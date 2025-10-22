# -------------------------------------------------------------
# Custom LiteLLM handler that forwards to a Gradio Space
# -------------------------------------------------------------
import os
import time
import requests
from typing import Iterator, AsyncIterator

from litellm import CustomLLM, completion, acompletion
from litellm.types.utils import GenericStreamingChunk, ModelResponse

# ------------------------------------------------------------------
# Helper: convert a Gradio SSE chunk into a LiteLLM GenericStreamingChunk
# ------------------------------------------------------------------
def _sse_to_chunk(text: str, index: int = 0, finished: bool = False) -> GenericStreamingChunk:
    return {
        "finish_reason": "stop" if finished else None,
        "index": index,
        "is_finished": finished,
        "text": text,
        "tool_use": None,
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
    }

# ------------------------------------------------------------------
# Main custom LLM class
# ------------------------------------------------------------------
class GradioLLM(CustomLLM):
    """
    A LiteLLM custom provider that proxies all calls to a Gradio Space.
    The handler accepts the same arguments that the Gradio `respond` function
    expects and forwards them to the space via the `/api/predict` endpoint.
    """

    def __init__(self, space_url: str):
        super().__init__()
        self.space_url = space_url.rstrip("/")
        self.session = requests.Session()
        # Optional: set a timeout so a stuck request doesn't block the proxy
        self.timeout = 120

    # ------------------------------------------------------------------
    # Internal helper: call the Gradio API (sync)
    # ------------------------------------------------------------------
    def _call_space(self, payload: dict) -> str:
        """
        Calls the Gradio Space and returns the assistant's text output.
        """
        url = f"{self.space_url}/api/predict"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # The ChatInterface returns a list with a single element – the assistant string
        return data["data"][0]

    # ------------------------------------------------------------------
    # Internal helper: call the Gradio API (streaming)
    # ------------------------------------------------------------------
    def _stream_space(self, payload: dict) -> Iterator[str]:
        """
        Yields partial assistant outputs as they arrive from the Gradio SSE stream.
        """
        url = f"{self.space_url}/api/predict"
        # The `stream=True` flag tells Gradio to use Server‑Sent Events
        resp = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
        resp.raise_for_status()

        buffer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            # Gradio SSE lines start with "data: "
            if line.startswith(b"data: "):
                chunk = line[6:].decode()
                if chunk == "[DONE]":
                    if buffer:
                        yield buffer
                    break
                # The chunk is a JSON array; the first element is the partial text
                try:
                    import json
                    payload = json.loads(chunk)
                    partial = payload[0]
                    buffer += partial
                    yield buffer
                except Exception:
                    # Fallback – just yield raw chunk
                    yield chunk

    # ------------------------------------------------------------------
    # Build the payload that matches the Gradio `respond` signature
    # ------------------------------------------------------------------
    def _build_payload(self, **kwargs) -> dict:
        """
        Extracts all parameters from kwargs and builds the payload for the Space.
        """
        # Default values – these match the defaults in the Gradio app
        message = kwargs.get("message") or kwargs.get("messages", [{"role": "user", "content": ""}])[0]["content"]
        history = kwargs.get("history", [])
        model = kwargs.get("model", "n8n_Qwen3-8B")
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        max_tokens = kwargs.get("max_tokens", 4096)
        ctx_size = kwargs.get("ctx_size", 32192)
        temperature = kwargs.get("temperature", 0.4)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 40)
        repeat_penalty = kwargs.get("repeat_penalty", 1.0)
        long_duration = kwargs.get("long_duration", False)

        return {
            "data": [
                message,
                history,
                model,
                system_message,
                max_tokens,
                ctx_size,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                long_duration,
            ],
            "fn_index": 0,
        }

    # ------------------------------------------------------------------
    # OpenAI‑compatible completion
    # ------------------------------------------------------------------
    def completion(self, *args, **kwargs) -> ModelResponse:
        """
        Implements `completion` for the sync OpenAI route.
        """
        payload = self._build_payload(**kwargs)
        assistant_text = self._call_space(payload)

        # LiteLLM will automatically wrap `mock_response` into a ModelResponse
        return completion(
            model="custom/gradio-llama",
            messages=[{"role": "user", "content": payload["data"][0]}],
            mock_response=assistant_text,
        )

    # ------------------------------------------------------------------
    # Async completion (LiteLLM proxy may call this)
    # ------------------------------------------------------------------
    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """
        Async wrapper – simply forwards to the sync implementation.
        """
        return self.completion(*args, **kwargs)

    # ------------------------------------------------------------------
    # Streaming – returns a generator of GenericStreamingChunk
    # ------------------------------------------------------------------
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """
        Streams the assistant output from the Gradio Space.
        Each chunk is sent as a GenericStreamingChunk to LiteLLM.
        """
        payload = self._build_payload(**kwargs)
        for partial in self._stream_space(payload):
            # The partial string may contain the whole assistant output so far.
            # We send each chunk as a new GenericStreamingChunk.
            chunk = _sse_to_chunk(partial, finished=False)
            yield chunk
        # Final chunk – mark as finished
        yield _sse_to_chunk("", finished=True)

    # ------------------------------------------------------------------
    # Async streaming – LiteLLM may call this
    # ------------------------------------------------------------------
    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """
        Async wrapper around the sync streaming method.
        """
        for chunk in self.streaming(*args, **kwargs):
            yield chunk

# ------------------------------------------------------------------
# Instantiate the handler – replace the URL with your own Space
# ------------------------------------------------------------------
# Example: https://my-gradio-space.hf.space
SPACE_URL = os.getenv("GRADIO_SPACE_URL", "https://jaysadatay-llama-3-1-8b-instruct-and-codestral-22b-v0-1.hf.space/")
gradio_llm = GradioLLM(space_url=SPACE_URL)
