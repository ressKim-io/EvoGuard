package handler

import (
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ressKim-io/EvoGuard/api-service/internal/usecase"
)

// ErrorResponse represents a structured error response
type ErrorResponse struct {
	StatusCode int
	Code       string
	Message    string
}

// MapUsecaseError maps usecase errors to HTTP error responses.
// It provides consistent error handling across all handlers.
func MapUsecaseError(err error) ErrorResponse {
	switch {
	case errors.Is(err, usecase.ErrBattleNotFound):
		return ErrorResponse{
			StatusCode: http.StatusNotFound,
			Code:       "NOT_FOUND",
			Message:    "battle not found",
		}
	case errors.Is(err, usecase.ErrBattleNotRunnable):
		return ErrorResponse{
			StatusCode: http.StatusConflict,
			Code:       "CONFLICT",
			Message:    "battle cannot accept rounds",
		}
	case errors.Is(err, usecase.ErrBattleCompleted):
		return ErrorResponse{
			StatusCode: http.StatusConflict,
			Code:       "CONFLICT",
			Message:    "battle already completed",
		}
	case errors.Is(err, usecase.ErrInvalidRequest):
		return ErrorResponse{
			StatusCode: http.StatusBadRequest,
			Code:       "INVALID_REQUEST",
			Message:    "invalid request",
		}
	default:
		return ErrorResponse{
			StatusCode: http.StatusInternalServerError,
			Code:       "INTERNAL_ERROR",
			Message:    "internal server error",
		}
	}
}

// HandleUsecaseError handles a usecase error by sending an appropriate HTTP response.
// It maps the error to an HTTP status and sends a JSON error response.
func HandleUsecaseError(c *gin.Context, err error) {
	errResp := MapUsecaseError(err)
	respondError(c, errResp.StatusCode, errResp.Code, errResp.Message)
}

// HandleInvalidUUID handles an invalid UUID parameter error.
func HandleInvalidUUID(c *gin.Context, paramName string) {
	respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid "+paramName)
}

// HandleInvalidRequest handles a generic invalid request error.
func HandleInvalidRequest(c *gin.Context, message string) {
	respondError(c, http.StatusBadRequest, "INVALID_REQUEST", message)
}
