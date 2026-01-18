package client

import (
	"context"

	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/service"
)

// MLClassifier adapts MLClient to the Classifier interface
type MLClassifier struct {
	client *MLClient
}

// NewMLClassifier creates a new MLClassifier
func NewMLClassifier(client *MLClient) service.Classifier {
	return &MLClassifier{client: client}
}

// Classify classifies a single text
func (c *MLClassifier) Classify(ctx context.Context, text, requestID string) (*service.ClassificationResult, error) {
	resp, err := c.client.Classify(ctx, text, requestID)
	if err != nil {
		return nil, err
	}

	score := resp.Result.Confidence
	if !resp.Result.IsToxic {
		score = 1 - score
	}

	return &service.ClassificationResult{
		Label:      resp.Result.Label,
		Score:      score,
		Confidence: resp.Result.Confidence,
	}, nil
}

// ClassifyBatch classifies multiple texts
func (c *MLClassifier) ClassifyBatch(ctx context.Context, texts []string, requestID string) ([]*service.ClassificationResult, error) {
	resp, err := c.client.ClassifyBatch(ctx, texts, requestID)
	if err != nil {
		return nil, err
	}

	results := make([]*service.ClassificationResult, len(resp.Results))
	for i, r := range resp.Results {
		score := r.Confidence
		if !r.IsToxic {
			score = 1 - score
		}
		results[i] = &service.ClassificationResult{
			Label:      r.Label,
			Score:      score,
			Confidence: r.Confidence,
		}
	}

	return results, nil
}
