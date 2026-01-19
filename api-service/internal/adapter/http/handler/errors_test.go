package handler

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"

	"github.com/ressKim-io/EvoGuard/api-service/internal/usecase"
)

func TestMapUsecaseError(t *testing.T) {
	tests := []struct {
		name               string
		err                error
		expectedStatusCode int
		expectedCode       string
		expectedMessage    string
	}{
		{
			name:               "battle not found",
			err:                usecase.ErrBattleNotFound,
			expectedStatusCode: http.StatusNotFound,
			expectedCode:       "NOT_FOUND",
			expectedMessage:    "battle not found",
		},
		{
			name:               "battle not runnable",
			err:                usecase.ErrBattleNotRunnable,
			expectedStatusCode: http.StatusConflict,
			expectedCode:       "CONFLICT",
			expectedMessage:    "battle cannot accept rounds",
		},
		{
			name:               "battle completed",
			err:                usecase.ErrBattleCompleted,
			expectedStatusCode: http.StatusConflict,
			expectedCode:       "CONFLICT",
			expectedMessage:    "battle already completed",
		},
		{
			name:               "invalid request",
			err:                usecase.ErrInvalidRequest,
			expectedStatusCode: http.StatusBadRequest,
			expectedCode:       "INVALID_REQUEST",
			expectedMessage:    "invalid request",
		},
		{
			name:               "unknown error",
			err:                errors.New("some unknown error"),
			expectedStatusCode: http.StatusInternalServerError,
			expectedCode:       "INTERNAL_ERROR",
			expectedMessage:    "internal server error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MapUsecaseError(tt.err)

			assert.Equal(t, tt.expectedStatusCode, result.StatusCode)
			assert.Equal(t, tt.expectedCode, result.Code)
			assert.Equal(t, tt.expectedMessage, result.Message)
		})
	}
}

func TestHandleUsecaseError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name               string
		err                error
		expectedStatusCode int
	}{
		{
			name:               "battle not found",
			err:                usecase.ErrBattleNotFound,
			expectedStatusCode: http.StatusNotFound,
		},
		{
			name:               "internal error",
			err:                errors.New("internal"),
			expectedStatusCode: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			HandleUsecaseError(c, tt.err)

			assert.Equal(t, tt.expectedStatusCode, w.Code)
		})
	}
}

func TestHandleInvalidUUID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	HandleInvalidUUID(c, "battle id")

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "invalid battle id")
}

func TestHandleInvalidRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	HandleInvalidRequest(c, "missing required field")

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "missing required field")
}
