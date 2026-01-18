package config

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoad(t *testing.T) {
	t.Run("loads default configuration", func(t *testing.T) {
		cfg, err := Load()

		assert.NoError(t, err)
		assert.NotNil(t, cfg)

		// Check server defaults
		assert.Equal(t, "0.0.0.0", cfg.Server.Host)
		assert.Equal(t, 8080, cfg.Server.Port)
		assert.Equal(t, "debug", cfg.Server.Mode)

		// Check database defaults
		assert.Equal(t, "localhost", cfg.Database.Host)
		assert.Equal(t, 5432, cfg.Database.Port)
		assert.Equal(t, "evoguard", cfg.Database.User)
		assert.Equal(t, "evoguard", cfg.Database.Password)
		assert.Equal(t, "evoguard", cfg.Database.DBName)
		assert.Equal(t, "disable", cfg.Database.SSLMode)

		// Check redis defaults
		assert.Equal(t, "localhost", cfg.Redis.Host)
		assert.Equal(t, 6379, cfg.Redis.Port)
		assert.Equal(t, "", cfg.Redis.Password)
		assert.Equal(t, 0, cfg.Redis.DB)

		// Check log defaults
		assert.Equal(t, "info", cfg.Log.Level)
		assert.Equal(t, "json", cfg.Log.Format)
	})

	t.Run("reads from environment variables", func(t *testing.T) {
		// Set environment variables
		os.Setenv("EVOGUARD_SERVER_PORT", "9090")
		os.Setenv("EVOGUARD_DATABASE_HOST", "db.example.com")
		os.Setenv("EVOGUARD_LOG_LEVEL", "debug")
		defer func() {
			os.Unsetenv("EVOGUARD_SERVER_PORT")
			os.Unsetenv("EVOGUARD_DATABASE_HOST")
			os.Unsetenv("EVOGUARD_LOG_LEVEL")
		}()

		cfg, err := Load()

		assert.NoError(t, err)
		assert.Equal(t, 9090, cfg.Server.Port)
		assert.Equal(t, "db.example.com", cfg.Database.Host)
		assert.Equal(t, "debug", cfg.Log.Level)
	})
}

func TestSetDefaults(t *testing.T) {
	// This is implicitly tested through Load()
	// but we can verify the defaults are reasonable
	cfg, err := Load()
	assert.NoError(t, err)

	// Verify sensible defaults
	assert.Greater(t, cfg.Server.Port, 0)
	assert.Greater(t, cfg.Database.Port, 0)
	assert.Greater(t, cfg.Redis.Port, 0)
}
