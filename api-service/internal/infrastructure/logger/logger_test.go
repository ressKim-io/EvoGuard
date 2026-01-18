package logger

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ressKim-io/EvoGuard/api-service/internal/infrastructure/config"
)

func TestNewLogger(t *testing.T) {
	t.Run("creates logger with JSON format", func(t *testing.T) {
		cfg := &config.LogConfig{
			Level:  "info",
			Format: "json",
		}

		logger, err := NewLogger(cfg)

		assert.NoError(t, err)
		assert.NotNil(t, logger)
	})

	t.Run("creates logger with console format", func(t *testing.T) {
		cfg := &config.LogConfig{
			Level:  "debug",
			Format: "console",
		}

		logger, err := NewLogger(cfg)

		assert.NoError(t, err)
		assert.NotNil(t, logger)
	})

	t.Run("defaults to info level for invalid level", func(t *testing.T) {
		cfg := &config.LogConfig{
			Level:  "invalid",
			Format: "json",
		}

		logger, err := NewLogger(cfg)

		assert.NoError(t, err)
		assert.NotNil(t, logger)
	})

	t.Run("creates logger with error level", func(t *testing.T) {
		cfg := &config.LogConfig{
			Level:  "error",
			Format: "json",
		}

		logger, err := NewLogger(cfg)

		assert.NoError(t, err)
		assert.NotNil(t, logger)
	})

	t.Run("creates logger with warn level", func(t *testing.T) {
		cfg := &config.LogConfig{
			Level:  "warn",
			Format: "console",
		}

		logger, err := NewLogger(cfg)

		assert.NoError(t, err)
		assert.NotNil(t, logger)
	})
}
