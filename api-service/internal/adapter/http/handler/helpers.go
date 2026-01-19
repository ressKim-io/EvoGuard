package handler

import (
	"fmt"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// PaginationParams holds pagination parameters
type PaginationParams struct {
	Limit  int
	Offset int
}

// Default pagination values
const (
	DefaultLimit  = 20
	MaxLimit      = 100
	DefaultOffset = 0
)

// ParsePagination extracts and validates pagination parameters from the request.
// It returns validated PaginationParams with safe default values.
func ParsePagination(c *gin.Context) *PaginationParams {
	limit, err := strconv.Atoi(c.DefaultQuery("limit", strconv.Itoa(DefaultLimit)))
	if err != nil || limit < 1 {
		limit = DefaultLimit
	}
	if limit > MaxLimit {
		limit = MaxLimit
	}

	offset, err := strconv.Atoi(c.DefaultQuery("offset", strconv.Itoa(DefaultOffset)))
	if err != nil || offset < 0 {
		offset = DefaultOffset
	}

	return &PaginationParams{
		Limit:  limit,
		Offset: offset,
	}
}

// ExtractUUIDParam extracts and parses a UUID parameter from the URL path.
// Returns the parsed UUID or an error if the parameter is invalid.
func ExtractUUIDParam(c *gin.Context, param string) (uuid.UUID, error) {
	idStr := c.Param(param)
	id, err := uuid.Parse(idStr)
	if err != nil {
		return uuid.Nil, fmt.Errorf("invalid %s: %w", param, err)
	}
	return id, nil
}

// ValidAttackStrategies contains all valid attack strategy values
var ValidAttackStrategies = map[string]bool{
	"unicode_evasion": true,
	"homoglyph":       true,
	"leetspeak":       true,
	"llm_evasion":     true,
	"adversarial_llm": true,
	"mixed":           true,
}

// IsValidAttackStrategy checks if the given strategy is valid
func IsValidAttackStrategy(strategy string) bool {
	return ValidAttackStrategies[strategy]
}
