package handler

import (
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// Response represents the standard API response structure
type Response struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   *ErrorInfo  `json:"error,omitempty"`
	Meta    *MetaInfo   `json:"meta"`
}

// ErrorInfo represents error details
type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// MetaInfo represents response metadata
type MetaInfo struct {
	Timestamp string `json:"timestamp"`
	RequestID string `json:"request_id"`
}

func newMeta(c *gin.Context) *MetaInfo {
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = uuid.New().String()
	}
	return &MetaInfo{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		RequestID: requestID,
	}
}

func respondSuccess(c *gin.Context, status int, data interface{}) {
	c.JSON(status, Response{
		Success: true,
		Data:    data,
		Meta:    newMeta(c),
	})
}

func respondError(c *gin.Context, status int, code, message string) {
	c.JSON(status, Response{
		Success: false,
		Error: &ErrorInfo{
			Code:    code,
			Message: message,
		},
		Meta: newMeta(c),
	})
}
