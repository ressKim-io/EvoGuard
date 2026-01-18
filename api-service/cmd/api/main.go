package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/ressKim-io/EvoGuard/api-service/internal/adapter/http/router"
	"github.com/ressKim-io/EvoGuard/api-service/internal/infrastructure/cache"
	"github.com/ressKim-io/EvoGuard/api-service/internal/infrastructure/config"
	"github.com/ressKim-io/EvoGuard/api-service/internal/infrastructure/database"
	"github.com/ressKim-io/EvoGuard/api-service/internal/infrastructure/logger"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Initialize logger
	log, err := logger.NewLogger(&cfg.Log)
	if err != nil {
		return fmt.Errorf("failed to initialize logger: %w", err)
	}
	defer func() { _ = log.Sync() }()

	// Set Gin mode
	gin.SetMode(cfg.Server.Mode)

	// Initialize database
	db, err := database.NewPostgresDB(&cfg.Database)
	if err != nil {
		log.Error("Failed to connect to database", zap.Error(err))
		return fmt.Errorf("failed to connect to database: %w", err)
	}
	log.Info("Connected to database")

	// Run migrations
	if err := database.AutoMigrate(db); err != nil {
		log.Error("Failed to run migrations", zap.Error(err))
		return fmt.Errorf("failed to run migrations: %w", err)
	}
	log.Info("Database migrations completed")

	// Initialize Redis (optional, continue without it)
	redisClient, err := cache.NewRedisClient(&cfg.Redis)
	if err != nil {
		log.Warn("Failed to connect to Redis, continuing without cache", zap.Error(err))
		redisClient = nil
	} else {
		log.Info("Connected to Redis")
	}

	// Setup router
	r := router.Setup(db, redisClient, log)

	// Create HTTP server
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Info("Starting server", zap.String("address", addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("Server failed", zap.Error(err))
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error("Server forced to shutdown", zap.Error(err))
	}

	// Close database connection
	if sqlDB, err := db.DB(); err == nil && sqlDB != nil {
		_ = sqlDB.Close()
	}

	// Close Redis connection
	if redisClient != nil {
		_ = redisClient.Close()
	}

	log.Info("Server exited")
	return nil
}
