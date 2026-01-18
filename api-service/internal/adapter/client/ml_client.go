package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ClassifyRequest represents a request to the ML service
type ClassifyRequest struct {
	Text      string `json:"text"`
	RequestID string `json:"request_id,omitempty"`
}

// ClassifyBatchRequest represents a batch request to the ML service
type ClassifyBatchRequest struct {
	Texts     []string `json:"texts"`
	RequestID string   `json:"request_id,omitempty"`
}

// ClassificationResult represents a single classification result
type ClassificationResult struct {
	Text       string  `json:"text"`
	IsToxic    bool    `json:"is_toxic"`
	Confidence float64 `json:"confidence"`
	Label      string  `json:"label"`
}

// ClassifyResponse represents the response from the ML service
type ClassifyResponse struct {
	Success      bool                 `json:"success"`
	Result       ClassificationResult `json:"result"`
	ModelVersion string               `json:"model_version"`
	RequestID    string               `json:"request_id,omitempty"`
}

// ClassifyBatchResponse represents the batch response from the ML service
type ClassifyBatchResponse struct {
	Success      bool                   `json:"success"`
	Results      []ClassificationResult `json:"results"`
	ModelVersion string                 `json:"model_version"`
	RequestID    string                 `json:"request_id,omitempty"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status       string `json:"status"`
	ModelLoaded  bool   `json:"model_loaded"`
	ModelVersion string `json:"model_version"`
}

// MLClient is an HTTP client for the ML service
type MLClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewMLClient creates a new ML service client
func NewMLClient(baseURL string, timeout time.Duration) *MLClient {
	return &MLClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// Classify sends a single text for classification
func (c *MLClient) Classify(ctx context.Context, text, requestID string) (*ClassifyResponse, error) {
	reqBody := ClassifyRequest{
		Text:      text,
		RequestID: requestID,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/classify", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
		}
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// ClassifyBatch sends multiple texts for classification
func (c *MLClient) ClassifyBatch(ctx context.Context, texts []string, requestID string) (*ClassifyBatchResponse, error) {
	reqBody := ClassifyBatchRequest{
		Texts:     texts,
		RequestID: requestID,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/classify/batch", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
		}
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result ClassifyBatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// Health checks the ML service health
func (c *MLClient) Health(ctx context.Context) (*HealthResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var result HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// Ready checks if the ML service is ready
func (c *MLClient) Ready(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/ready", http.NoBody)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ML service not ready: status %d", resp.StatusCode)
	}

	return nil
}
