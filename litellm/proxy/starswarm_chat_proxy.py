import json
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid


class StarSwarmProxyError(Exception):
    def __init__(self, status_code: int, error: dict):
        super().__init__(error.get("message", "StarSwarm proxy error"))
        self.status_code = status_code
        self.error = error


def is_starswarm_chat_proxy_enabled() -> bool:
    return bool(_get_scheduler_base_url())


def _get_scheduler_base_url() -> str:
    return os.getenv("STARSWARM_SCHEDULER_BASE_URL", "").rstrip("/")


def _get_timeout_seconds() -> float:
    raw_value = os.getenv("STARSWARM_SCHEDULER_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw_value)
    except ValueError:
        timeout = 120.0
    return timeout if timeout > 0 else 120.0


def _build_trace_id(request: Request) -> str:
    return request.headers.get("X-Trace-ID") or str(uuid.uuid4())


def _build_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


def _build_error_response(
    status_code: int,
    code: str,
    message: str,
    trace_id: Optional[str] = None,
) -> JSONResponse:
    error: Dict[str, Any] = {"code": code, "message": message}
    if trace_id:
        error["trace_id"] = trace_id
    return JSONResponse(status_code=status_code, content={"error": error})


def _map_error_code_to_status(error_code: Optional[str]) -> int:
    if error_code == "AUTH_INVALID_API_KEY":
        return status.HTTP_401_UNAUTHORIZED
    if error_code == "BILLING_INSUFFICIENT_BALANCE":
        return status.HTTP_402_PAYMENT_REQUIRED
    if error_code == "MODEL_NOT_FOUND":
        return status.HTTP_404_NOT_FOUND
    if error_code == "UNSUPPORTED_MEDIA_TYPE":
        return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    if error_code == "RATE_LIMITED":
        return status.HTTP_429_TOO_MANY_REQUESTS
    if error_code == "REQUEST_VALIDATION_FAILED":
        return status.HTTP_400_BAD_REQUEST
    return status.HTTP_503_SERVICE_UNAVAILABLE


def _extract_openai_error(body: Any, fallback_status: int) -> dict:
    if isinstance(body, dict):
        if isinstance(body.get("error"), dict):
            error = dict(body["error"])
            code = error.get("code")
            if not code:
                error["code"] = "TASK_AGENT_FAILED"
            return error

        data = body.get("data")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            error = dict(data["error"])
            code = error.get("code")
            if not code:
                error["code"] = "TASK_AGENT_FAILED"
            return error

        message = body.get("msg") or body.get("message") or "StarSwarm request failed"
        code = body.get("error_code") or body.get("code") or "TASK_AGENT_FAILED"
        error: Dict[str, Any] = {"code": str(code), "message": str(message)}
        trace_id = body.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            error["trace_id"] = trace_id
        return error

    return {
        "code": "TASK_AGENT_FAILED",
        "message": f"StarSwarm request failed with HTTP {fallback_status}",
    }


def _build_scheduler_request_body(data: dict) -> dict:
    payload: Dict[str, Any] = {
        "model": data.get("model"),
        "messages": data.get("messages"),
        "stream": bool(data.get("stream")),
    }
    if data.get("max_tokens") is not None:
        payload["max_tokens"] = data.get("max_tokens")
    return payload


async def _post_auth_and_route(
    request: Request, data: dict
) -> tuple[dict, str, str]:
    scheduler_base_url = _get_scheduler_base_url()
    if not scheduler_base_url:
        raise RuntimeError("STARSWARM_SCHEDULER_BASE_URL is not configured")

    authorization = request.headers.get("Authorization")
    if not authorization:
        raise StarSwarmProxyError(
            status.HTTP_401_UNAUTHORIZED,
            {
                "code": "AUTH_INVALID_API_KEY",
                "message": "Missing Authorization header",
            },
        )

    trace_id = _build_trace_id(request)
    request_id = _build_request_id(request)
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "X-Trace-ID": trace_id,
        "X-Request-ID": request_id,
    }
    payload = _build_scheduler_request_body(data)
    timeout = httpx.Timeout(_get_timeout_seconds())

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{scheduler_base_url}/internal/v1/auth-and-route",
            headers=headers,
            json=payload,
        )

    try:
        body = response.json()
    except json.JSONDecodeError:
        body = None

    if response.status_code >= 400:
        error = _extract_openai_error(body, response.status_code)
        raise StarSwarmProxyError(response.status_code, error)

    if not isinstance(body, dict):
        raise StarSwarmProxyError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "code": "TASK_AGENT_FAILED",
                "message": "StarSwarm auth-and-route returned a non-JSON response",
                "trace_id": trace_id,
            },
        )

    auth_data = body.get("data")
    if not isinstance(auth_data, dict) or not auth_data.get("taskId"):
        error = _extract_openai_error(body, response.status_code)
        raise StarSwarmProxyError(_map_error_code_to_status(error.get("code")), error)

    return auth_data, trace_id, request_id


def _build_stream_headers(trace_id: str) -> dict:
    return {"X-Trace-ID": trace_id}


def _scheduler_stream_url(task_id: str) -> str:
    return f"{_get_scheduler_base_url()}/internal/v1/tasks/{task_id}/stream"


def _extract_chunk_content(chunk_payload: dict) -> str:
    choices = chunk_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    delta = first_choice.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _extract_finish_reason(chunk_payload: dict) -> Optional[str]:
    choices = chunk_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    finish_reason = first_choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) and finish_reason else None


async def _read_nonstream_chat_completion(
    request: Request,
    task_id: str,
    trace_id: str,
    requested_model: str,
) -> dict:
    stream_url = _scheduler_stream_url(task_id)
    timeout = httpx.Timeout(connect=_get_timeout_seconds(), read=None, write=None, pool=None)
    contents: list[str] = []
    finish_reason: Optional[str] = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "GET",
            stream_url,
            headers=_build_stream_headers(trace_id),
        ) as response:
            if response.status_code >= 400:
                try:
                    body = await response.aread()
                    parsed_body = json.loads(body.decode("utf-8"))
                except Exception:
                    parsed_body = None
                error = _extract_openai_error(parsed_body, response.status_code)
                raise StarSwarmProxyError(response.status_code, error)

            async for line in response.aiter_lines():
                if await request.is_disconnected():
                    raise StarSwarmProxyError(
                        499,
                        {
                            "code": "TASK_CLIENT_CANCELED",
                            "message": "Client disconnected the request",
                            "trace_id": trace_id,
                        },
                    )

                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk_payload = json.loads(data_str)
                except json.JSONDecodeError:
                    verbose_proxy_logger.warning(
                        "StarSwarm non-stream parser skipped invalid SSE chunk task_id=%s chunk=%s",
                        task_id,
                        data_str,
                    )
                    continue

                if isinstance(chunk_payload, dict) and isinstance(
                    chunk_payload.get("error"), dict
                ):
                    error = dict(chunk_payload["error"])
                    raise StarSwarmProxyError(
                        _map_error_code_to_status(error.get("code")),
                        error,
                    )

                chunk_content = _extract_chunk_content(chunk_payload)
                if chunk_content:
                    contents.append(chunk_content)

                chunk_finish_reason = _extract_finish_reason(chunk_payload)
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason

    return {
        "id": task_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(contents)},
                "finish_reason": finish_reason or "stop",
            }
        ],
    }


def _build_scheduler_stream_error_frame(error: dict) -> bytes:
    payload = json.dumps({"error": error}, ensure_ascii=False)
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode("utf-8")


async def _scheduler_stream_generator(
    request: Request,
    task_id: str,
    trace_id: str,
) -> AsyncGenerator[bytes, None]:
    stream_url = _scheduler_stream_url(task_id)
    timeout = httpx.Timeout(connect=_get_timeout_seconds(), read=None, write=None, pool=None)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "GET",
                stream_url,
                headers=_build_stream_headers(trace_id),
            ) as response:
                if response.status_code >= 400:
                    try:
                        body = await response.aread()
                        parsed_body = json.loads(body.decode("utf-8"))
                    except Exception:
                        parsed_body = None
                    error = _extract_openai_error(parsed_body, response.status_code)
                    yield _build_scheduler_stream_error_frame(error)
                    return

                async for chunk in response.aiter_raw():
                    if await request.is_disconnected():
                        verbose_proxy_logger.info(
                            "StarSwarm upstream stream closed after client disconnect task_id=%s trace_id=%s",
                            task_id,
                            trace_id,
                        )
                        return
                    yield chunk
        except httpx.HTTPError as exc:
            verbose_proxy_logger.exception(
                "StarSwarm upstream stream failed task_id=%s trace_id=%s",
                task_id,
                trace_id,
            )
            error = {
                "code": "TASK_AGENT_FAILED",
                "message": f"StarSwarm upstream stream failed: {exc}",
                "trace_id": trace_id,
            }
            yield _build_scheduler_stream_error_frame(error)


async def maybe_handle_starswarm_chat_completion(
    request: Request,
    data: dict,
) -> Optional[JSONResponse | StreamingResponse]:
    if not is_starswarm_chat_proxy_enabled():
        return None

    auth_data, trace_id, _request_id = await _post_auth_and_route(request, data)
    task_id = auth_data["taskId"]
    requested_model = (
        auth_data.get("publicModelName")
        or data.get("model")
        or auth_data.get("backendModelName")
        or "starswarm-auto"
    )

    if bool(data.get("stream")):
        return StreamingResponse(
            _scheduler_stream_generator(request=request, task_id=task_id, trace_id=trace_id),
            media_type="text/event-stream",
            headers={"X-Trace-ID": trace_id},
        )

    completion_response = await _read_nonstream_chat_completion(
        request=request,
        task_id=task_id,
        trace_id=trace_id,
        requested_model=requested_model,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=completion_response,
        headers={"X-Trace-ID": trace_id},
    )
