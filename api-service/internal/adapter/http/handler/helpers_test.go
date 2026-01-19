package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestParsePagination(t *testing.T) {
	tests := []struct {
		name           string
		queryParams    map[string]string
		expectedLimit  int
		expectedOffset int
	}{
		{
			name:           "default values",
			queryParams:    map[string]string{},
			expectedLimit:  DefaultLimit,
			expectedOffset: DefaultOffset,
		},
		{
			name:           "custom valid values",
			queryParams:    map[string]string{"limit": "50", "offset": "10"},
			expectedLimit:  50,
			expectedOffset: 10,
		},
		{
			name:           "limit exceeds max",
			queryParams:    map[string]string{"limit": "200"},
			expectedLimit:  MaxLimit,
			expectedOffset: DefaultOffset,
		},
		{
			name:           "negative limit uses default",
			queryParams:    map[string]string{"limit": "-5"},
			expectedLimit:  DefaultLimit,
			expectedOffset: DefaultOffset,
		},
		{
			name:           "negative offset uses default",
			queryParams:    map[string]string{"offset": "-5"},
			expectedLimit:  DefaultLimit,
			expectedOffset: DefaultOffset,
		},
		{
			name:           "invalid limit uses default",
			queryParams:    map[string]string{"limit": "invalid"},
			expectedLimit:  DefaultLimit,
			expectedOffset: DefaultOffset,
		},
		{
			name:           "invalid offset uses default",
			queryParams:    map[string]string{"offset": "invalid"},
			expectedLimit:  DefaultLimit,
			expectedOffset: DefaultOffset,
		},
		{
			name:           "zero limit uses default",
			queryParams:    map[string]string{"limit": "0"},
			expectedLimit:  DefaultLimit,
			expectedOffset: DefaultOffset,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			req := httptest.NewRequest(http.MethodGet, "/", nil)
			q := req.URL.Query()
			for k, v := range tt.queryParams {
				q.Add(k, v)
			}
			req.URL.RawQuery = q.Encode()
			c.Request = req

			pagination := ParsePagination(c)

			assert.Equal(t, tt.expectedLimit, pagination.Limit)
			assert.Equal(t, tt.expectedOffset, pagination.Offset)
		})
	}
}

func TestExtractUUIDParam(t *testing.T) {
	tests := []struct {
		name       string
		paramValue string
		expectErr  bool
	}{
		{
			name:       "valid UUID",
			paramValue: "550e8400-e29b-41d4-a716-446655440000",
			expectErr:  false,
		},
		{
			name:       "invalid UUID",
			paramValue: "invalid-uuid",
			expectErr:  true,
		},
		{
			name:       "empty string",
			paramValue: "",
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Params = gin.Params{{Key: "id", Value: tt.paramValue}}

			id, err := ExtractUUIDParam(c, "id")

			if tt.expectErr {
				assert.Error(t, err)
				assert.Equal(t, uuid.Nil, id)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.paramValue, id.String())
			}
		})
	}
}

func TestIsValidAttackStrategy(t *testing.T) {
	validStrategies := []string{
		"unicode_evasion",
		"homoglyph",
		"leetspeak",
		"llm_evasion",
		"adversarial_llm",
		"mixed",
	}

	invalidStrategies := []string{
		"invalid",
		"",
		"UNICODE_EVASION",
		"unknown_strategy",
	}

	for _, strategy := range validStrategies {
		t.Run("valid_"+strategy, func(t *testing.T) {
			assert.True(t, IsValidAttackStrategy(strategy))
		})
	}

	for _, strategy := range invalidStrategies {
		t.Run("invalid_"+strategy, func(t *testing.T) {
			assert.False(t, IsValidAttackStrategy(strategy))
		})
	}
}
