package handler

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"gorm.io/gorm"
)

// HealthHandler handles health check endpoints
type HealthHandler struct {
	db    *gorm.DB
	redis *redis.Client
}

// NewHealthHandler creates a new health handler
func NewHealthHandler(db *gorm.DB, redis *redis.Client) *HealthHandler {
	return &HealthHandler{
		db:    db,
		redis: redis,
	}
}

// HealthStatus represents the health check response
type HealthStatus struct {
	Status     string            `json:"status"`
	Components map[string]string `json:"components"`
}

// Health handles GET /health
func (h *HealthHandler) Health(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	components := make(map[string]string)
	healthy := true

	// Check database
	if h.db != nil {
		sqlDB, err := h.db.DB()
		if err != nil {
			components["database"] = "error: " + err.Error()
			healthy = false
		} else if err := sqlDB.PingContext(ctx); err != nil {
			components["database"] = "error: " + err.Error()
			healthy = false
		} else {
			components["database"] = "ok"
		}
	} else {
		components["database"] = "not configured"
	}

	// Check Redis
	if h.redis != nil {
		if err := h.redis.Ping(ctx).Err(); err != nil {
			components["redis"] = "error: " + err.Error()
			healthy = false
		} else {
			components["redis"] = "ok"
		}
	} else {
		components["redis"] = "not configured"
	}

	status := "healthy"
	httpStatus := http.StatusOK
	if !healthy {
		status = "unhealthy"
		httpStatus = http.StatusServiceUnavailable
	}

	c.JSON(httpStatus, HealthStatus{
		Status:     status,
		Components: components,
	})
}

// Ready handles GET /ready
func (h *HealthHandler) Ready(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	// Check database connection
	if h.db != nil {
		sqlDB, err := h.db.DB()
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "not ready", "reason": "database error"})
			return
		}
		if err := sqlDB.PingContext(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "not ready", "reason": "database unreachable"})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{"status": "ready"})
}
