package service

import "context"

// ClassificationResult represents the result of text classification
type ClassificationResult struct {
	Label      string  `json:"label"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
}

// Classifier defines the interface for text classification
type Classifier interface {
	// Classify classifies a single text
	Classify(ctx context.Context, text, requestID string) (*ClassificationResult, error)

	// ClassifyBatch classifies multiple texts
	ClassifyBatch(ctx context.Context, texts []string, requestID string) ([]*ClassificationResult, error)
}
