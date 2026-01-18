package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMLClient_Classify(t *testing.T) {
	t.Run("successful classification", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/classify", r.URL.Path)
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

			var req ClassifyRequest
			err := json.NewDecoder(r.Body).Decode(&req)
			require.NoError(t, err)
			assert.Equal(t, "test text", req.Text)
			assert.Equal(t, "req-123", req.RequestID)

			resp := ClassifyResponse{
				Success: true,
				Result: ClassificationResult{
					Text:       "test text",
					IsToxic:    false,
					Confidence: 0.85,
					Label:      "non-toxic",
				},
				ModelVersion: "mock-v1.0.0",
				RequestID:    "req-123",
			}
			w.Header().Set("Content-Type", "application/json")
			err = json.NewEncoder(w).Encode(resp)
			require.NoError(t, err)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		result, err := client.Classify(context.Background(), "test text", "req-123")

		require.NoError(t, err)
		assert.True(t, result.Success)
		assert.Equal(t, "test text", result.Result.Text)
		assert.False(t, result.Result.IsToxic)
		assert.Equal(t, 0.85, result.Result.Confidence)
		assert.Equal(t, "mock-v1.0.0", result.ModelVersion)
	})

	t.Run("server error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, err := w.Write([]byte("internal error"))
			require.NoError(t, err)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		_, err := client.Classify(context.Background(), "test", "")

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "500")
	})

	t.Run("connection error", func(t *testing.T) {
		client := NewMLClient("http://localhost:99999", 1*time.Second)
		_, err := client.Classify(context.Background(), "test", "")

		assert.Error(t, err)
	})
}

func TestMLClient_ClassifyBatch(t *testing.T) {
	t.Run("successful batch classification", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/classify/batch", r.URL.Path)

			var req ClassifyBatchRequest
			err := json.NewDecoder(r.Body).Decode(&req)
			require.NoError(t, err)
			assert.Len(t, req.Texts, 2)

			resp := ClassifyBatchResponse{
				Success: true,
				Results: []ClassificationResult{
					{Text: "text1", IsToxic: false, Confidence: 0.9, Label: "non-toxic"},
					{Text: "text2", IsToxic: true, Confidence: 0.8, Label: "toxic"},
				},
				ModelVersion: "mock-v1.0.0",
			}
			w.Header().Set("Content-Type", "application/json")
			err = json.NewEncoder(w).Encode(resp)
			require.NoError(t, err)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		result, err := client.ClassifyBatch(context.Background(), []string{"text1", "text2"}, "")

		require.NoError(t, err)
		assert.True(t, result.Success)
		assert.Len(t, result.Results, 2)
		assert.False(t, result.Results[0].IsToxic)
		assert.True(t, result.Results[1].IsToxic)
	})
}

func TestMLClient_Health(t *testing.T) {
	t.Run("healthy service", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/health", r.URL.Path)
			assert.Equal(t, "GET", r.Method)

			resp := HealthResponse{
				Status:       "healthy",
				ModelLoaded:  true,
				ModelVersion: "mock-v1.0.0",
			}
			w.Header().Set("Content-Type", "application/json")
			err := json.NewEncoder(w).Encode(resp)
			require.NoError(t, err)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		result, err := client.Health(context.Background())

		require.NoError(t, err)
		assert.Equal(t, "healthy", result.Status)
		assert.True(t, result.ModelLoaded)
	})
}

func TestMLClient_Ready(t *testing.T) {
	t.Run("ready service", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/ready", r.URL.Path)
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		err := client.Ready(context.Background())

		assert.NoError(t, err)
	})

	t.Run("not ready service", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		err := client.Ready(context.Background())

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not ready")
	})
}
