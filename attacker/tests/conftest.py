"""Pytest fixtures for attacker tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from attacker.ollama_client import GenerateResponse, OllamaClient, OllamaSettings


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return [
        "hello world",
        "This is a test",
        "바보",
        "안녕하세요",
        "Hello123",
        "UPPERCASE",
        "MixedCase Text",
    ]


@pytest.fixture
def korean_texts() -> list[str]:
    """Korean sample texts."""
    return [
        "바보",
        "안녕하세요",
        "테스트입니다",
        "한글 텍스트",
    ]


@pytest.fixture
def english_texts() -> list[str]:
    """English sample texts."""
    return [
        "hello",
        "world",
        "test",
        "HELLO",
        "MixedCase",
    ]


@pytest.fixture
def ollama_settings() -> OllamaSettings:
    """Test Ollama settings."""
    return OllamaSettings(
        base_url="http://localhost:11434",
        model="mistral:7b-instruct-v0.2-q4_K_S",
        timeout=30.0,
        max_retries=1,
    )


@pytest.fixture
def mock_ollama_response() -> GenerateResponse:
    """Mock Ollama response."""
    return GenerateResponse(
        model="mistral:7b-instruct-v0.2-q4_K_S",
        response="transformed test text",
        done=True,
        total_duration=1000000,
        eval_count=10,
    )


@pytest.fixture
def mock_ollama_client(mock_ollama_response: GenerateResponse) -> MagicMock:
    """Mock Ollama client."""
    client = MagicMock(spec=OllamaClient)
    client.generate = AsyncMock(return_value=mock_ollama_response)
    client.health_check = AsyncMock(return_value=True)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client
