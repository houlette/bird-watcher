# Common dev tasks for BirdWatcher. Run `make` or `make help` to see what's available.
.DEFAULT_GOAL := help

# The remote host we deploy to. Override at the CLI if it ever changes.
VM ?= ryan@birdwatcher.ryanhoulette.com

.PHONY: help test build smoke calibrate vapid deploy logs lint fmt fe-build fe-dev clean clean-db

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1;36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ─── Local development ────────────────────────────────────────────────────────

test: ## Run the backend pytest suite (no ML deps needed for 37/37)
	cd backend && python3 -m pytest tests/ -v

build: ## Build the api Docker image
	docker compose build api

smoke: ## End-to-end smoke test: run pipeline against backend/data/clips/smoke_test.webm
	docker compose run --rm api python scripts/smoke_test.py

fe-build: ## Build the frontend (TypeScript + Vite) into frontend/dist
	cd frontend && npm ci --no-audit --no-fund && npm run build

fe-dev: ## Run the frontend dev server with HMR (assumes api running on localhost:8000)
	cd frontend && npm run dev

lint: ## ruff lint check (treat as advisory)
	cd backend && python3 -m ruff check . || true

fmt: ## ruff format (in-place)
	cd backend && python3 -m ruff format .

# ─── One-shot setup / maintenance ─────────────────────────────────────────────

vapid: ## Generate VAPID keys (one-time). Prints public key for backend/.env
	docker compose run --rm api python scripts/generate_vapid_keys.py

calibrate: ## Refresh yard_priors.json from Haikubox API (last 200 days)
	docker compose run --rm api python scripts/calibrate_from_haikubox.py --days 200

retrain-check: ## Print accumulated active-learning corrections (stub; doesn't retrain yet)
	docker compose run --rm api python scripts/retrain_classifier.py

# ─── Production deploy ────────────────────────────────────────────────────────

deploy: ## rsync the repo to $(VM) and run bootstrap_server.sh there
	bash scripts/deploy_to_server.sh $(VM)

logs: ## Tail production API logs
	ssh $(VM) 'cd ~/BirdWatcher && docker compose logs -f --tail=50 api'

logs-caddy: ## Tail production Caddy (TLS / HTTP routing) logs
	ssh $(VM) 'cd ~/BirdWatcher && docker compose logs -f --tail=50 caddy'

remote-shell: ## SSH into the VM with cwd in the project root
	ssh -t $(VM) 'cd ~/BirdWatcher && bash'

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove pycache, test artifacts, frontend node_modules
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf backend/.pytest_cache
	rm -rf frontend/node_modules frontend/dist

clean-db: ## Wipe local SQLite + crops (forces fresh smoke run)
	rm -f backend/data/birdwatcher.db
	find backend/data/crops -name '*.jpg' -delete 2>/dev/null || true
