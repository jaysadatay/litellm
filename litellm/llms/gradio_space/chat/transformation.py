import os
import json
import httpx
from typing import Dict, Any, Iterator, AsyncIterator

from litellm.llms.base import BaseConfig
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.custom_httpx.llm_http_handler import CustomStreamWrapper

# ------------------------------------------------------------------
# 1.  Configuration class – this is the “adapter” LiteLLM will call
# ------------------------------------------------------------------
class GradioSpaceChat(BaseConfig):
    """
    A custom provider that forwards OpenAI‑style chat requests to a Gradio Space.
    """

    # ---- 1.1  Basic metadata ---------------------------------------
    name: str = "gradio-llama"          # value you will use in the `model` field
    provider: str = "gradio_space"      # unique provider key
    # The URL of the space – can be overridden via env var
    space_url: str = os.getenv("GRADIO_SPACE_URL", "https://jaysadatay-llama-3-1-8b-instruct-and-codestral-22b-v0-1.hf.space")

    # ---- 1.2  Request helpers ---------------------------------------
    def _build_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Convert the OpenAI request into the payload that the Gradio
        `/api/predict` endpoint expects.
        """
        # Grab the values that the Gradio `respond` function uses.
        # The defaults mirror those in the original space code.
        message = kwargs.get("message") or kwargs.get("messages", [{"role": "user", "content": ""}])[0]["content"]
        history = kwargs.get("history", [])
        model = kwargs.get("model", "Meta-Llama-3.1-8B-Instruct.Q6_K.gguf")
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        max_tokens = kwargs.get("max_tokens", 4096)
        ctx_size = kwargs.get("ctx_size", 8192)
        temperature = kwargs.get("temperature", 0.1)
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
            "fn_index": 0,          # the ChatInterface is the first tab
        }

    # ------------------------------------------------------------------
    # 2.  HTTP helpers – sync & async
    # ------------------------------------------------------------------
    def _post(self, payload: Dict[str, Any], stream: bool = False) -> httpx.Response:
        url = f"{self.space_url}/api/predict"
        client = httpx.Client(timeout=120)
        return client.post(url, json=payload, stream=stream)

    async def _apost(self, payload: Dict[str, Any], stream: bool = False) -> httpx.Response:
        url = f"{self.space_url}/api/predict"
        async with httpx.AsyncClient(timeout=120) as ac:
            return await ac.post(url, json=payload, stream=stream)

    # ------------------------------------------------------------------
    # 3.  Core methods required by BaseConfig
    # ------------------------------------------------------------------
    def validate_environment(self, **kwargs) -> None:
        # no special validation needed – the space URL is hard‑coded
        pass

    def get_complete_url(self, **kwargs) -> str:
        # The endpoint is fixed – just return it
        return f"{self.space_url}/api/predict"

    def transform_request(self, **kwargs) -> Dict[str, Any]:
        # Convert the OpenAI request into the Gradio payload
        return self._build_payload(**kwargs)

    def transform_response(self, response: httpx.Response, **kwargs) -> ModelResponse:
        """
        Convert the raw JSON from Gradio into a LiteLLM ModelResponse.
        """
        raw = response.json()
        # Gradio returns {"data": ["assistant text"]} – we only care about the first element
        assistant_text = raw["data"][0]
        return ModelResponse(
            model=self.name,
            choices=[
                {
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # ------------------------------------------------------------------
    # 4.  Streaming helpers
    # ------------------------------------------------------------------
    def _sse_to_chunk(self, text: str, finished: bool = False) -> GenericStreamingChunk:
        return {
            "finish_reason": "stop" if finished else None,
            "index": 0,
            "is_finished": finished,
            "text": text,
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }

    def get_sync_custom_stream_wrapper(self, **kwargs) -> Iterator[GenericStreamingChunk]:
        """
        Called by LiteLLM when `stream=True` is passed to a sync request.
        """
        payload = self.transform_request(**kwargs)
        resp = self._post(payload, stream=True)
        resp.raise_for_status()

        buffer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                chunk = line[6:].decode()
                if chunk == "[DONE]":
                    if buffer:
                        yield self._sse_to_chunk(buffer, finished=True)
                    break
                try:
                    data = json.loads(chunk)
                    partial = data[0]  # the partial assistant text
                    buffer += partial
                    yield self._sse_to_chunk(buffer, finished=False)
                except Exception:
                    yield self._sse_to_chunk(chunk, finished=False)

    async def get_async_custom_stream_wrapper(self, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """
        Async counterpart – yields chunks as they arrive.
        """
        payload = self.transform_request(**kwargs)
        resp = await self._apost(payload, stream=True)
        resp.raise_for_status()

        buffer = ""
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                chunk = line[6:].decode()
                if chunk == "[DONE]":
                    if buffer:
                        yield self._sse_to_chunk(buffer, finished=True)
                    break
                try:
                    data = json.loads(chunk)
                    partial = data[0]
                    buffer += partial
                    yield self._sse_to_chunk(buffer, finished=False)
                except Exception:
                    yield self._sse_to_chunk(chunk, finished=False)
