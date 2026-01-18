package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestMLClassifier_Classify(t *testing.T) {
	t.Run("toxic text returns positive score", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/classify", r.URL.Path)
			assert.Equal(t, "POST", r.Method)

			resp := ClassifyResponse{
				Success: true,
				Result: ClassificationResult{
					Text:       "toxic text",
					IsToxic:    true,
					Confidence: 0.95,
					Label:      "toxic",
				},
				ModelVersion: "mock-v1",
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		classifier := NewMLClassifier(client)

		result, err := classifier.Classify(context.Background(), "toxic text", "test-request-id")

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "toxic", result.Label)
		assert.Equal(t, 0.95, result.Confidence)
		assert.Equal(t, 0.95, result.Score)
	})

	t.Run("non-toxic text returns inverted score", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			resp := ClassifyResponse{
				Success: true,
				Result: ClassificationResult{
					Text:       "hello world",
					IsToxic:    false,
					Confidence: 0.2,
					Label:      "non-toxic",
				},
				ModelVersion: "mock-v1",
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		classifier := NewMLClassifier(client)

		result, err := classifier.Classify(context.Background(), "hello world", "test-request-id")

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "non-toxic", result.Label)
		assert.Equal(t, 0.2, result.Confidence)
		assert.Equal(t, 0.8, result.Score) // 1 - confidence for non-toxic
	})

	t.Run("server error returns error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("internal error"))
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		classifier := NewMLClassifier(client)

		result, err := classifier.Classify(context.Background(), "text", "test-request-id")

		assert.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestMLClassifier_ClassifyBatch(t *testing.T) {
	t.Run("batch classification success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/classify/batch", r.URL.Path)
			assert.Equal(t, "POST", r.Method)

			resp := ClassifyBatchResponse{
				Success: true,
				Results: []ClassificationResult{
					{Text: "text1", IsToxic: true, Confidence: 0.9, Label: "toxic"},
					{Text: "text2", IsToxic: false, Confidence: 0.1, Label: "non-toxic"},
				},
				ModelVersion: "mock-v1",
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		classifier := NewMLClassifier(client)

		results, err := classifier.ClassifyBatch(context.Background(), []string{"text1", "text2"}, "test-request-id")

		assert.NoError(t, err)
		assert.Len(t, results, 2)
		assert.Equal(t, "toxic", results[0].Label)
		assert.Equal(t, 0.9, results[0].Score)
		assert.Equal(t, "non-toxic", results[1].Label)
		assert.Equal(t, 0.9, results[1].Score) // 1 - 0.1 = 0.9
	})

	t.Run("batch classification error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer server.Close()

		client := NewMLClient(server.URL, 5*time.Second)
		classifier := NewMLClassifier(client)

		results, err := classifier.ClassifyBatch(context.Background(), []string{"text1", "text2"}, "test-request-id")

		assert.Error(t, err)
		assert.Nil(t, results)
	})
}
