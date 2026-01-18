-- =============================================================================
-- EvoGuard PostgreSQL Initialization
-- =============================================================================

-- Create MLflow database
CREATE DATABASE mlflow;

-- Create application user (optional, for production)
-- CREATE USER evoguard_app WITH PASSWORD 'your_password';
-- GRANT ALL PRIVILEGES ON DATABASE evoguard TO evoguard_app;
-- GRANT ALL PRIVILEGES ON DATABASE mlflow TO evoguard_app;

-- Extensions for main database
\c evoguard;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Extensions for MLflow database
\c mlflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
