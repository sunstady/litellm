import importlib.util
import logging
import pathlib
import sys
import types


def _load_starswarm_chat_proxy_module():
    module_name = "tests_starswarm_chat_proxy_module"
    module_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "litellm"
        / "proxy"
        / "starswarm_chat_proxy.py"
    )

    litellm_module = types.ModuleType("litellm")
    litellm_logging_module = types.ModuleType("litellm._logging")
    litellm_logging_module.verbose_proxy_logger = logging.getLogger(
        "tests.starswarm_chat_proxy"
    )
    litellm_uuid_module = types.ModuleType("litellm._uuid")
    litellm_uuid_module.uuid = lambda: "test-uuid"

    sys.modules.setdefault("litellm", litellm_module)
    sys.modules["litellm._logging"] = litellm_logging_module
    sys.modules["litellm._uuid"] = litellm_uuid_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_should_map_public_error_codes_to_expected_status():
    module = _load_starswarm_chat_proxy_module()

    assert module._map_error_code_to_status("AUTH_INVALID_API_KEY") == 401
    assert module._map_error_code_to_status("BILLING_INSUFFICIENT_BALANCE") == 402
    assert module._map_error_code_to_status("MODEL_NOT_FOUND") == 404
    assert module._map_error_code_to_status("RATE_LIMITED") == 429
    assert module._map_error_code_to_status("TASK_AGENT_FAILED") == 503


def test_should_extract_error_from_openai_shape():
    module = _load_starswarm_chat_proxy_module()
    body = {
        "error": {
            "code": "TASK_AGENT_FAILED",
            "message": "agent failed",
            "trace_id": "trace-1",
        }
    }

    assert module._extract_openai_error(body, 503) == body["error"]


def test_should_extract_error_from_common_result_shape():
    module = _load_starswarm_chat_proxy_module()
    body = {
        "code": 1234,
        "msg": "balance is insufficient",
        "trace_id": "trace-2",
    }

    error = module._extract_openai_error(body, 402)

    assert error["code"] == "1234"
    assert error["message"] == "balance is insufficient"
    assert error["trace_id"] == "trace-2"


def test_should_extract_delta_content_and_finish_reason():
    module = _load_starswarm_chat_proxy_module()
    chunk = {
        "id": "task-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": None,
            }
        ],
    }
    finish_chunk = {
        "id": "task-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    assert module._extract_chunk_content(chunk) == "hello"
    assert module._extract_finish_reason(chunk) is None
    assert module._extract_finish_reason(finish_chunk) == "stop"
