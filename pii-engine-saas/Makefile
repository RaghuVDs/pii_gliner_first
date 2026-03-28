# =============================================================================
# PII Engine SaaS — Makefile
# =============================================================================

COMPOSE        := docker compose
COMPOSE_DEV    := $(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml
BACKEND_EXEC   := $(COMPOSE) exec backend
PYTEST_OPTS    ?=

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

.PHONY: dev
dev: ## Start all services with hot-reload (dev mode)
	$(COMPOSE_DEV) up --build

.PHONY: dev-detached
dev-detached: ## Start all services in background (dev mode)
	$(COMPOSE_DEV) up --build -d

.PHONY: logs
logs: ## Tail logs from all services
	$(COMPOSE) logs -f

# -----------------------------------------------------------------------------
# Build & Deploy
# -----------------------------------------------------------------------------

.PHONY: build
build: ## Build all Docker images
	$(COMPOSE) build

.PHONY: up
up: ## Start all services (production compose)
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop all services (preserve volumes)
	$(COMPOSE) down

.PHONY: restart
restart: down up ## Restart all services

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------

.PHONY: migrate
migrate: ## Run Alembic migrations (upgrade to head)
	$(BACKEND_EXEC) alembic upgrade head

.PHONY: migrate-generate
migrate-generate: ## Auto-generate a new Alembic migration (usage: make migrate-generate MSG="add users table")
	$(BACKEND_EXEC) alembic revision --autogenerate -m "$(MSG)"

.PHONY: migrate-downgrade
migrate-downgrade: ## Downgrade one migration
	$(BACKEND_EXEC) alembic downgrade -1

.PHONY: seed
seed: ## Run database seed script
	$(BACKEND_EXEC) python -m app.scripts.seed

# -----------------------------------------------------------------------------
# Testing & Quality
# -----------------------------------------------------------------------------

.PHONY: test
test: ## Run backend tests with pytest
	$(BACKEND_EXEC) pytest $(PYTEST_OPTS) tests/ -v --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(BACKEND_EXEC) pytest tests/ -v --tb=short --cov=app --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Lint backend code with ruff
	$(BACKEND_EXEC) ruff check app/ tests/

.PHONY: lint-fix
lint-fix: ## Auto-fix lint issues
	$(BACKEND_EXEC) ruff check --fix app/ tests/

.PHONY: format
format: ## Format backend code with ruff
	$(BACKEND_EXEC) ruff format app/ tests/

.PHONY: typecheck
typecheck: ## Run mypy type checking
	$(BACKEND_EXEC) mypy app/ --ignore-missing-imports

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

.PHONY: clean
clean: ## Stop all services and remove volumes
	$(COMPOSE) down -v --remove-orphans

.PHONY: clean-all
clean-all: clean ## Remove volumes, images, and orphan containers
	$(COMPOSE) down -v --rmi all --remove-orphans

.PHONY: prune
prune: ## Remove dangling Docker resources
	docker system prune -f

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

.PHONY: shell
shell: ## Open a bash shell in the backend container
	$(BACKEND_EXEC) bash

.PHONY: dbshell
dbshell: ## Open a psql shell to PostgreSQL
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-pii_user} -d $${POSTGRES_DB:-pii_engine}

.PHONY: env
env: ## Copy .env.example to .env (will not overwrite existing .env)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env created from .env.example — edit it with your secrets."; \
	else \
		echo ".env already exists, skipping."; \
	fi

.PHONY: help
help: ## Show this help message
	@echo "PII Engine SaaS — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
