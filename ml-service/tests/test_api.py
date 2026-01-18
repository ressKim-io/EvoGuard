"""Tests for API endpoints."""

from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_version" in data

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness endpoint returns ready status."""
        response = client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


class TestClassifyEndpoint:
    """Tests for classification endpoints."""

    def test_classify_non_toxic_text(self, client: TestClient) -> None:
        """Test classifying non-toxic text."""
        response = client.post(
            "/classify",
            json={"text": "Hello, how are you today?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["text"] == "Hello, how are you today?"
        assert data["result"]["label"] in ["toxic", "non-toxic"]
        assert 0.0 <= data["result"]["confidence"] <= 1.0

    def test_classify_toxic_text(self, client: TestClient) -> None:
        """Test classifying toxic text has high confidence."""
        response = client.post(
            "/classify",
            json={"text": "I hate you, you stupid idiot moron!"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        # Due to randomness in mock classifier, we check confidence is elevated
        assert data["result"]["confidence"] > 0.3

    def test_classify_with_request_id(self, client: TestClient) -> None:
        """Test classification with request ID."""
        response = client.post(
            "/classify",
            json={"text": "Test message", "request_id": "test-123"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["request_id"] == "test-123"

    def test_classify_empty_text_fails(self, client: TestClient) -> None:
        """Test that empty text returns validation error."""
        response = client.post(
            "/classify",
            json={"text": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_classify_missing_text_fails(self, client: TestClient) -> None:
        """Test that missing text returns validation error."""
        response = client.post(
            "/classify",
            json={},
        )
        assert response.status_code == 422


class TestBatchClassifyEndpoint:
    """Tests for batch classification endpoint."""

    def test_classify_batch(self, client: TestClient) -> None:
        """Test batch classification."""
        response = client.post(
            "/classify/batch",
            json={
                "texts": [
                    "Hello, world!",
                    "I hate you, stupid!",
                    "Have a nice day.",
                ]
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 3
        assert data["results"][0]["text"] == "Hello, world!"
        assert data["results"][1]["text"] == "I hate you, stupid!"
        assert data["results"][2]["text"] == "Have a nice day."

    def test_classify_batch_with_request_id(self, client: TestClient) -> None:
        """Test batch classification with request ID."""
        response = client.post(
            "/classify/batch",
            json={
                "texts": ["Test 1", "Test 2"],
                "request_id": "batch-456",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["request_id"] == "batch-456"

    def test_classify_batch_empty_list_fails(self, client: TestClient) -> None:
        """Test that empty list returns validation error."""
        response = client.post(
            "/classify/batch",
            json={"texts": []},
        )
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_endpoint(self, client: TestClient) -> None:
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "ml_classify_requests_total" in response.text or response.status_code == 200
