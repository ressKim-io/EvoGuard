package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestHealthHandler_Health(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("healthy when no dependencies", func(t *testing.T) {
		handler := NewHealthHandler(nil, nil)

		router := gin.New()
		router.GET("/health", handler.Health)

		req, _ := http.NewRequest("GET", "/health", http.NoBody)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var status HealthStatus
		err := json.Unmarshal(w.Body.Bytes(), &status)
		assert.NoError(t, err)
		assert.Equal(t, "healthy", status.Status)
		assert.Equal(t, "not configured", status.Components["database"])
		assert.Equal(t, "not configured", status.Components["redis"])
	})
}

func TestHealthHandler_Ready(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("ready when no database", func(t *testing.T) {
		handler := NewHealthHandler(nil, nil)

		router := gin.New()
		router.GET("/ready", handler.Ready)

		req, _ := http.NewRequest("GET", "/ready", http.NoBody)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), "ready")
	})
}
