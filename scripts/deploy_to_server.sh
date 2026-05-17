#!/usr/bin/env bash
# deploy_to_server.sh — runs FROM your Mac to push the current repo state
# (including the gitignored .env) to a remote BirdWatcher VM and invoke
# scripts/bootstrap_server.sh there.
#
# Usage:
#   scripts/deploy_to_server.sh ryan@birdwatcher.ryanhoulette.com
#
# Idempotent — re-runs sync changes and rebuild whatever changed.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <user>@<host>" >&2
  exit 2
fi

TARGET="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

if [ ! -f backend/.env ]; then
  echo "Refusing to deploy: backend/.env is missing. Create it from .env.example first." >&2
  exit 1
fi

echo "Syncing repo to $TARGET:~/BirdWatcher/ …"
# Notes on exclusions:
#   - .git is intentionally synced so the remote can see commit hashes if needed
#   - node_modules and .venv are huge and host-specific; rebuilt on the remote
#   - backend/data and backend/models are runtime artifacts; remote builds them
#   - frontend/dist is rebuilt on remote
ssh "$TARGET" "mkdir -p ~/BirdWatcher"
rsync -avz --delete-after \
  --exclude='.venv/' \
  --exclude='node_modules/' \
  --exclude='backend/data/' \
  --exclude='backend/models/' \
  --exclude='frontend/dist/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  ./ "$TARGET:~/BirdWatcher/"

echo "Invoking bootstrap on $TARGET …"
ssh "$TARGET" "cd ~/BirdWatcher && bash scripts/bootstrap_server.sh"
echo "Deploy complete."
