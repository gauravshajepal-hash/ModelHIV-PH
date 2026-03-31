from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_MODEL_ID = os.environ.get("EPIGRAPH_LIGHTON_OCR_MODEL", "lightonai/LightOnOCR-2-1B")
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", str(ROOT / ".hf-cache"))
DEFAULT_HOST = os.environ.get("EPIGRAPH_LIGHTON_OCR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("EPIGRAPH_LIGHTON_OCR_PORT", "8010"))
DEFAULT_DTYPE = os.environ.get("EPIGRAPH_LIGHTON_OCR_DTYPE", "float16")
DEFAULT_DEVICE = os.environ.get(
    "EPIGRAPH_LIGHTON_OCR_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("EPIGRAPH_LIGHTON_OCR_MAX_NEW_TOKENS", "3072"))


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = 1.0


app = FastAPI(title="LightOn OCR Local Endpoint", version="0.1.0")
_infer_lock = threading.Lock()
_state: dict[str, Any] = {
    "ready": False,
    "model_id": DEFAULT_MODEL_ID,
    "device": DEFAULT_DEVICE,
    "dtype_name": DEFAULT_DTYPE,
    "processor": None,
    "model": None,
    "startup_error": None,
    "request_count": 0,
    "last_latency_seconds": None,
}


def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    lowered = str(dtype_name or "").strip().lower()
    if lowered == "bfloat16" and device.startswith("cuda"):
        return torch.bfloat16
    if lowered == "float16" and device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _materialize_image_url(url: str) -> tuple[str, str | None]:
    normalized = str(url or "").strip()
    if normalized.startswith("data:image/"):
        header, encoded = normalized.split(",", 1)
        extension = ".png"
        if ";base64" not in header:
            raise ValueError("Unsupported data URL encoding")
        if "image/jpeg" in header:
            extension = ".jpg"
        elif "image/webp" in header:
            extension = ".webp"
        temp_dir = Path(tempfile.gettempdir()) / "epigraph_lighton_ocr"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4().hex}{extension}"
        temp_path.write_bytes(base64.b64decode(encoded))
        return str(temp_path), str(temp_path)
    return normalized, None


def _normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    conversation: list[dict[str, Any]] = []
    temp_paths: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        raw_content = message.get("content") or []
        if isinstance(raw_content, str):
            raw_content = [{"type": "text", "text": raw_content}]
        parts: list[dict[str, Any]] = []
        for item in raw_content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in {"text", "input_text"}:
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append({"type": "text", "text": text})
                continue
            if item_type in {"image", "image_url", "input_image"}:
                image_value = item.get("image_url")
                if isinstance(image_value, dict):
                    image_value = image_value.get("url")
                if image_value is None:
                    image_value = item.get("url")
                image_url = str(image_value or "").strip()
                if not image_url:
                    continue
                materialized_url, temp_path = _materialize_image_url(image_url)
                if temp_path:
                    temp_paths.append(temp_path)
                parts.append({"type": "image", "url": materialized_url})
        if parts:
            conversation.append({"role": role, "content": parts})
    return conversation, temp_paths


def _gpu_stats() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "memory_allocated_bytes": 0,
            "memory_reserved_bytes": 0,
            "max_memory_allocated_bytes": 0,
            "device_name": None,
        }
    return {
        "cuda_available": True,
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "device_name": torch.cuda.get_device_name(0),
    }


def _load_model() -> None:
    dtype = _resolve_dtype(DEFAULT_DTYPE, DEFAULT_DEVICE)
    processor = LightOnOcrProcessor.from_pretrained(DEFAULT_MODEL_ID, cache_dir=DEFAULT_CACHE_DIR)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_ID,
        cache_dir=DEFAULT_CACHE_DIR,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(DEFAULT_DEVICE)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _state.update(
        {
            "ready": True,
            "processor": processor,
            "model": model,
            "dtype_name": str(dtype).replace("torch.", ""),
            "device": DEFAULT_DEVICE,
            "startup_error": None,
        }
    )


@app.on_event("startup")
def _startup() -> None:
    try:
        _load_model()
    except Exception as exc:  # pragma: no cover
        _state["startup_error"] = f"{exc.__class__.__name__}: {exc}"
        _state["ready"] = False
        raise


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _state["ready"] else "error",
        "ready": bool(_state["ready"]),
        "model_id": _state["model_id"],
        "device": _state["device"],
        "dtype": _state["dtype_name"],
        "request_count": int(_state["request_count"]),
        "last_latency_seconds": _state["last_latency_seconds"],
        "startup_error": _state["startup_error"],
        "cache_dir": DEFAULT_CACHE_DIR,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "gpu": _gpu_stats(),
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _state["model_id"],
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["startup_error"] or "Model not ready")
    conversation, temp_paths = _normalize_messages(request.messages)
    if not any(
        part.get("type") == "image"
        for message in conversation
        for part in message.get("content", [])
        if isinstance(part, dict)
    ):
        raise HTTPException(status_code=400, detail="OCR request requires at least one image")
    processor = _state["processor"]
    model = _state["model"]
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model state unavailable")
    max_new_tokens = max(256, min(int(request.max_tokens or DEFAULT_MAX_NEW_TOKENS), DEFAULT_MAX_NEW_TOKENS))
    start = time.time()
    try:
        with _infer_lock:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            moved_inputs: dict[str, Any] = {}
            dtype = _resolve_dtype(_state["dtype_name"], str(_state["device"]))
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if getattr(value, "is_floating_point", lambda: False)():
                        moved_inputs[key] = value.to(device=_state["device"], dtype=dtype)
                    else:
                        moved_inputs[key] = value.to(device=_state["device"])
                else:
                    moved_inputs[key] = value
            with torch.inference_mode():
                output_ids = model.generate(
                    **moved_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            prompt_tokens = int(moved_inputs["input_ids"].shape[1])
            generated_ids = output_ids[0, prompt_tokens:]
            output_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            latency = round(time.time() - start, 4)
            _state["request_count"] = int(_state["request_count"]) + 1
            _state["last_latency_seconds"] = latency
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": _state["model_id"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(generated_ids.shape[-1]),
                    "total_tokens": prompt_tokens + int(generated_ids.shape[-1]),
                },
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc
    finally:
        for temp_path in temp_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                continue


@app.post("/v1/responses")
def responses(request: ChatCompletionRequest) -> dict[str, Any]:
    chat_payload = chat_completions(request)
    return {
        "id": f"resp-{uuid.uuid4().hex}",
        "object": "response",
        "model": chat_payload.get("model"),
        "output_text": chat_payload.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": chat_payload.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    }
                ],
            }
        ],
        "usage": chat_payload.get("usage", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
