# EvoGuard (Content Arena) Makefile
# Usage: make help

# ============================================================================
# Variables
# ============================================================================

# Directories
GO_SERVICE := api-service
PY_SERVICES := ml-service attacker defender training mlops
INFRA_DIR := infra

# Tools
GO := go
PYTHON := python3
UV := uv
DOCKER_COMPOSE := docker compose

# Build
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
GO_LDFLAGS := -ldflags="-X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)"

# Environment
ENV ?= development

# Colors
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
CYAN   := $(shell tput -Txterm setaf 6)
RESET  := $(shell tput -Txterm sgr0)

# ============================================================================
# Default & Help
# ============================================================================

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo ''
	@echo '$(CYAN)EvoGuard (Content Arena)$(RESET)'
	@echo ''
	@echo 'Usage: make $(GREEN)<target>$(RESET)'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ''

# ============================================================================
# Setup
# ============================================================================

.PHONY: setup
setup: setup-hooks setup-go setup-python setup-env ## Initial project setup (run once)
	@echo "$(GREEN)Setup complete!$(RESET)"

.PHONY: setup-hooks
setup-hooks: ## Install git hooks
	@echo "$(CYAN)Installing git hooks...$(RESET)"
	@bash scripts/setup-hooks.sh

.PHONY: setup-go
setup-go: ## Install Go dependencies
	@echo "$(CYAN)Installing Go dependencies...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) mod download; \
	fi

.PHONY: setup-python
setup-python: ## Install Python dependencies (using uv)
	@echo "$(CYAN)Installing Python dependencies...$(RESET)"
	@for service in $(PY_SERVICES); do \
		if [ -d "$$service" ] && [ -f "$$service/pyproject.toml" ]; then \
			echo "  Installing $$service..."; \
			cd $$service && $(UV) sync && cd ..; \
		fi; \
	done

.PHONY: setup-env
setup-env: ## Copy .env.example to .env if not exists
	@if [ ! -f .env ] && [ -f .env.example ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env from .env.example$(RESET)"; \
	fi

# ============================================================================
# Lint
# ============================================================================

.PHONY: lint
lint: go-lint py-lint ## Run all linters

.PHONY: go-lint
go-lint: ## Run Go linter (golangci-lint)
	@echo "$(CYAN)[Go] Running golangci-lint...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && golangci-lint run ./...; \
	else \
		echo "$(YELLOW)$(GO_SERVICE) not found, skipping$(RESET)"; \
	fi

.PHONY: py-lint
py-lint: ## Run Python linter (ruff)
	@echo "$(CYAN)[Python] Running ruff...$(RESET)"
	@for service in $(PY_SERVICES); do \
		if [ -d "$$service" ]; then \
			echo "  Checking $$service..."; \
			cd $$service && ruff check . && cd ..; \
		fi; \
	done

.PHONY: py-format
py-format: ## Format Python code (ruff format)
	@echo "$(CYAN)[Python] Formatting with ruff...$(RESET)"
	@for service in $(PY_SERVICES); do \
		if [ -d "$$service" ]; then \
			cd $$service && ruff format . && cd ..; \
		fi; \
	done

.PHONY: py-type
py-type: ## Run Python type checker (mypy)
	@echo "$(CYAN)[Python] Running mypy...$(RESET)"
	@for service in $(PY_SERVICES); do \
		if [ -d "$$service" ] && [ -f "$$service/pyproject.toml" ]; then \
			cd $$service && mypy . && cd ..; \
		fi; \
	done

# ============================================================================
# Test
# ============================================================================

.PHONY: test
test: go-test py-test ## Run all tests

.PHONY: go-test
go-test: ## Run Go tests
	@echo "$(CYAN)[Go] Running tests...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) test -v -race ./...; \
	else \
		echo "$(YELLOW)$(GO_SERVICE) not found, skipping$(RESET)"; \
	fi

.PHONY: go-test-cover
go-test-cover: ## Run Go tests with coverage
	@echo "$(CYAN)[Go] Running tests with coverage...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) test -v -race -coverprofile=coverage.out ./... && \
		$(GO) tool cover -html=coverage.out -o coverage.html; \
	fi

.PHONY: py-test
py-test: ## Run Python tests (pytest)
	@echo "$(CYAN)[Python] Running pytest...$(RESET)"
	@for service in $(PY_SERVICES); do \
		if [ -d "$$service" ]; then \
			echo "  Testing $$service..."; \
			cd $$service && pytest -v && cd ..; \
		fi; \
	done

.PHONY: py-test-cover
py-test-cover: ## Run Python tests with coverage
	@echo "$(CYAN)[Python] Running pytest with coverage...$(RESET)"
	@for service in $(PY_SERVICES); do \
		if [ -d "$$service" ]; then \
			cd $$service && pytest --cov=. --cov-report=html && cd ..; \
		fi; \
	done

# ============================================================================
# Build
# ============================================================================

.PHONY: build
build: go-build ## Build all services

.PHONY: go-build
go-build: ## Build Go binary
	@echo "$(CYAN)[Go] Building...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) build $(GO_LDFLAGS) -o bin/api-service ./cmd/api; \
	else \
		echo "$(YELLOW)$(GO_SERVICE) not found, skipping$(RESET)"; \
	fi

# ============================================================================
# Run
# ============================================================================

.PHONY: run
run: ## Run all services (docker-compose)
	@echo "$(CYAN)Starting services...$(RESET)"
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml up

.PHONY: run-go
run-go: ## Run Go API service locally
	@echo "$(CYAN)[Go] Running API service...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) run ./cmd/api; \
	fi

.PHONY: run-ml
run-ml: ## Run ML service locally
	@echo "$(CYAN)[Python] Running ML service...$(RESET)"
	@if [ -d "ml-service" ]; then \
		cd ml-service && $(UV) run uvicorn app.main:app --reload --port 8001; \
	fi

# ============================================================================
# Docker
# ============================================================================

.PHONY: docker-build
docker-build: ## Build all Docker images
	@echo "$(CYAN)Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml build

.PHONY: docker-up
docker-up: ## Start all containers (detached)
	@echo "$(CYAN)Starting containers...$(RESET)"
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml up -d

.PHONY: docker-down
docker-down: ## Stop all containers
	@echo "$(CYAN)Stopping containers...$(RESET)"
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml down

.PHONY: docker-logs
docker-logs: ## Show container logs
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml logs -f

.PHONY: docker-ps
docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml ps

# ============================================================================
# Database
# ============================================================================

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(CYAN)Running migrations...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) run ./cmd/migrate up; \
	fi

.PHONY: db-rollback
db-rollback: ## Rollback last migration
	@echo "$(CYAN)Rolling back migration...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		cd $(GO_SERVICE) && $(GO) run ./cmd/migrate down; \
	fi

# ============================================================================
# MLOps
# ============================================================================

.PHONY: mlflow-ui
mlflow-ui: ## Start MLflow UI
	@echo "$(CYAN)Starting MLflow UI...$(RESET)"
	mlflow ui --host 0.0.0.0 --port 5000

.PHONY: train
train: ## Run model training
	@echo "$(CYAN)Starting training...$(RESET)"
	@if [ -d "training" ]; then \
		cd training && $(UV) run python train.py; \
	fi

# ============================================================================
# Clean
# ============================================================================

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning...$(RESET)"
	@if [ -d "$(GO_SERVICE)" ]; then \
		rm -rf $(GO_SERVICE)/bin $(GO_SERVICE)/coverage.*; \
	fi
	@for service in $(PY_SERVICES); do \
		rm -rf $$service/.pytest_cache $$service/.mypy_cache $$service/.ruff_cache; \
		rm -rf $$service/htmlcov $$service/.coverage; \
		find $$service -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true; \
	done
	@echo "$(GREEN)Clean complete!$(RESET)"

.PHONY: clean-docker
clean-docker: ## Remove all Docker resources
	@echo "$(CYAN)Cleaning Docker resources...$(RESET)"
	$(DOCKER_COMPOSE) -f $(INFRA_DIR)/docker-compose.yml down -v --rmi local

# ============================================================================
# CI Helpers
# ============================================================================

.PHONY: ci-lint
ci-lint: ## CI: Run linters (fail on error)
	@$(MAKE) lint

.PHONY: ci-test
ci-test: ## CI: Run tests with coverage
	@$(MAKE) go-test-cover py-test-cover

.PHONY: ci
ci: ci-lint ci-test build ## CI: Full pipeline
	@echo "$(GREEN)CI pipeline complete!$(RESET)"
