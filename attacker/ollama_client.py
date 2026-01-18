"""Ollama HTTP client for LLM-based evasion strategies."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class OllamaSettings(BaseSettings):
    """Ollama client configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        extra="ignore",
    )

    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b-instruct-v0.2-q4_K_S"
    timeout: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    default_temperature: float = 0.9
    default_max_tokens: int = 256


class GenerateRequest(BaseModel):
    """Request model for Ollama generate endpoint."""

    model: str
    prompt: str
    stream: bool = False
    options: dict[str, Any] = field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response model from Ollama generate endpoint."""

    model: str
    response: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@dataclass
class OllamaError(Exception):
    """Ollama API error."""

    message: str
    status_code: int | None = None
    response_body: str | None = None


class OllamaClient:
    """Async HTTP client for Ollama API.

    Features:
    - Exponential backoff retry
    - Configurable timeout
    - Structured request/response handling
    """

    def __init__(self, settings: OllamaSettings | None = None) -> None:
        """Initialize client.

        Args:
            settings: Optional settings override
        """
        self._settings = settings or OllamaSettings()
        self._client: httpx.AsyncClient | None = None

    @property
    def settings(self) -> OllamaSettings:
        return self._settings

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=httpx.Timeout(self._settings.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerateResponse:
        """Generate text completion from Ollama.

        Args:
            prompt: Input prompt
            model: Model override (default from settings)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences

        Returns:
            GenerateResponse with generated text

        Raises:
            OllamaError: On API errors
        """
        if not self._client:
            raise OllamaError("Client not initialized. Use async context manager.")

        options: dict[str, Any] = {
            "temperature": temperature or self._settings.default_temperature,
            "num_predict": max_tokens or self._settings.default_max_tokens,
        }

        if stop_sequences:
            options["stop"] = stop_sequences

        request_body = {
            "model": model or self._settings.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        return await self._request_with_retry(request_body)

    async def _request_with_retry(
        self, request_body: dict[str, Any]
    ) -> GenerateResponse:
        """Execute request with exponential backoff retry."""
        assert self._client is not None, "Client not initialized"
        last_error: Exception | None = None

        for attempt in range(self._settings.max_retries):
            try:
                response = await self._client.post("/api/generate", json=request_body)

                if response.status_code == 200:
                    return GenerateResponse(**response.json())

                # Non-retryable status codes
                if response.status_code in (400, 404):
                    raise OllamaError(
                        f"Request failed: {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                # Retryable errors
                last_error = OllamaError(
                    f"Request failed: {response.status_code}",
                    status_code=response.status_code,
                )

            except httpx.RequestError as e:
                last_error = OllamaError(f"Connection error: {e}")
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

            # Exponential backoff
            if attempt < self._settings.max_retries - 1:
                delay = self._settings.retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise last_error or OllamaError("Unknown error after retries")

    async def health_check(self) -> bool:
        """Check if Ollama server is available.

        Returns:
            True if server is healthy
        """
        if not self._client:
            raise OllamaError("Client not initialized. Use async context manager.")

        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names
        """
        if not self._client:
            raise OllamaError("Client not initialized. Use async context manager.")

        response = await self._client.get("/api/tags")
        if response.status_code != 200:
            raise OllamaError(
                "Failed to list models",
                status_code=response.status_code,
            )

        data = response.json()
        return [model["name"] for model in data.get("models", [])]


# Synchronous wrapper for non-async contexts
class SyncOllamaClient:
    """Synchronous wrapper for OllamaClient."""

    def __init__(self, settings: OllamaSettings | None = None) -> None:
        self._settings = settings or OllamaSettings()

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerateResponse:
        """Synchronous generate."""
        return asyncio.run(self._async_generate(prompt, model, temperature, max_tokens))

    async def _async_generate(
        self,
        prompt: str,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerateResponse:
        async with OllamaClient(self._settings) as client:
            return await client.generate(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def health_check(self) -> bool:
        """Synchronous health check."""
        return asyncio.run(self._async_health_check())

    async def _async_health_check(self) -> bool:
        async with OllamaClient(self._settings) as client:
            return await client.health_check()
