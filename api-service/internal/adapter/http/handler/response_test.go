package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestRespondSuccess(t *testing.T) {
	t.Run("returns success response with data", func(t *testing.T) {
		router := gin.New()
		router.GET("/test", func(c *gin.Context) {
			c.Set("request_id", "test-request-id")
			respondSuccess(c, http.StatusOK, map[string]string{"key": "value"})
		})

		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var response Response
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.True(t, response.Success)
		assert.NotNil(t, response.Data)
		assert.Nil(t, response.Error)
		assert.NotNil(t, response.Meta)
		assert.Equal(t, "test-request-id", response.Meta.RequestID)
	})

	t.Run("returns success response with created status", func(t *testing.T) {
		router := gin.New()
		router.POST("/test", func(c *gin.Context) {
			respondSuccess(c, http.StatusCreated, map[string]int{"id": 1})
		})

		req, _ := http.NewRequest("POST", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusCreated, w.Code)

		var response Response
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.True(t, response.Success)
	})
}

func TestRespondError(t *testing.T) {
	t.Run("returns error response", func(t *testing.T) {
		router := gin.New()
		router.GET("/test", func(c *gin.Context) {
			c.Set("request_id", "test-request-id")
			respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid input")
		})

		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response Response
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.False(t, response.Success)
		assert.Nil(t, response.Data)
		assert.NotNil(t, response.Error)
		assert.Equal(t, "INVALID_REQUEST", response.Error.Code)
		assert.Equal(t, "invalid input", response.Error.Message)
	})

	t.Run("returns 500 error response", func(t *testing.T) {
		router := gin.New()
		router.GET("/test", func(c *gin.Context) {
			respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "something went wrong")
		})

		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusInternalServerError, w.Code)

		var response Response
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.False(t, response.Success)
		assert.Equal(t, "INTERNAL_ERROR", response.Error.Code)
	})

	t.Run("generates request ID if not set", func(t *testing.T) {
		router := gin.New()
		router.GET("/test", func(c *gin.Context) {
			respondError(c, http.StatusNotFound, "NOT_FOUND", "resource not found")
		})

		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		var response Response
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.NotEmpty(t, response.Meta.RequestID)
	})
}

func TestNewMeta(t *testing.T) {
	t.Run("uses existing request ID", func(t *testing.T) {
		router := gin.New()
		router.GET("/test", func(c *gin.Context) {
			c.Set("request_id", "existing-id")
			meta := newMeta(c)
			c.JSON(http.StatusOK, meta)
		})

		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		var meta MetaInfo
		err := json.Unmarshal(w.Body.Bytes(), &meta)
		assert.NoError(t, err)
		assert.Equal(t, "existing-id", meta.RequestID)
		assert.NotEmpty(t, meta.Timestamp)
	})

	t.Run("generates new request ID when not set", func(t *testing.T) {
		router := gin.New()
		router.GET("/test", func(c *gin.Context) {
			meta := newMeta(c)
			c.JSON(http.StatusOK, meta)
		})

		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		var meta MetaInfo
		err := json.Unmarshal(w.Body.Bytes(), &meta)
		assert.NoError(t, err)
		assert.NotEmpty(t, meta.RequestID)
	})
}
