package router

import (
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
	"gorm.io/gorm"

	"github.com/ressKim-io/EvoGuard/api-service/internal/adapter/http/handler"
	"github.com/ressKim-io/EvoGuard/api-service/internal/adapter/http/middleware"
	"github.com/ressKim-io/EvoGuard/api-service/internal/adapter/repository/postgres"
	"github.com/ressKim-io/EvoGuard/api-service/internal/usecase"
)

// Setup creates and configures the Gin router
func Setup(db *gorm.DB, redisClient *redis.Client, logger *zap.Logger) *gin.Engine {
	router := gin.New()

	// Middleware
	router.Use(middleware.RequestID())
	router.Use(middleware.Logger(logger))
	router.Use(middleware.Recovery(logger))
	router.Use(middleware.CORS())

	// Health endpoints
	healthHandler := handler.NewHealthHandler(db, redisClient)
	router.GET("/health", healthHandler.Health)
	router.GET("/ready", healthHandler.Ready)

	// Prometheus metrics
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Initialize repositories
	battleRepo := postgres.NewBattleRepository(db)
	roundRepo := postgres.NewRoundRepository(db)

	// Initialize usecases
	battleUC := usecase.NewBattleUsecase(battleRepo, roundRepo)

	// Initialize handlers
	battleHandler := handler.NewBattleHandler(battleUC)

	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// Battle routes
		battles := v1.Group("/battles")
		{
			battles.POST("", battleHandler.CreateBattle)
			battles.GET("", battleHandler.ListBattles)
			battles.GET("/:id", battleHandler.GetBattle)
			battles.GET("/:id/stats", battleHandler.GetBattleStats)
			battles.POST("/:id/stop", battleHandler.StopBattle)
		}
	}

	return router
}
