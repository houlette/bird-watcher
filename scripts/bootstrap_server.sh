#!/usr/bin/env bash
# bootstrap_server.sh — runs ON the VM to take it from "fresh + hardened" to
# "BirdWatcher serving traffic." Idempotent: re-running is safe and skips
# anything already done.
#
# Steps 4-9 of DEPLOY.md, collapsed into one command. Steps 10-12 (camera
# config + phone install) are inherently manual and still need you.
#
# Run via:  scripts/deploy_to_server.sh <user>@<host>
# (that wrapper syncs the repo + .env up here, then invokes this script)

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

require_env() {
  if [ ! -f backend/.env ]; then
    die "backend/.env not found. deploy_to_server.sh should have synced it; \
investigate why it didn't."
  fi
}

# ────────────────────────────────────────────────────────────────────────────
# 1. Docker
# ────────────────────────────────────────────────────────────────────────────
install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker already installed: $(docker --version)"
    return
  fi
  log "Installing Docker (Ubuntu official repo)…"
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
  fi
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER" || true
  log "Docker installed. You may need to log out/in for the docker group to take effect."
}

# ────────────────────────────────────────────────────────────────────────────
# 2. Node 20 (for the Vite build)
# ────────────────────────────────────────────────────────────────────────────
install_node() {
  if command -v node >/dev/null 2>&1 && [ "$(node -v | cut -dv -f2 | cut -d. -f1)" -ge 20 ]; then
    log "Node $(node -v) already installed."
    return
  fi
  log "Installing Node 20…"
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y -qq nodejs
  log "Node $(node -v) installed."
}

# ────────────────────────────────────────────────────────────────────────────
# 3. Build the API image (first build is slow; later runs are cached)
# ────────────────────────────────────────────────────────────────────────────
build_api() {
  log "Building API Docker image (first time: ~5-10 min for torch + transformers + ultralytics)…"
  # Need sudo if user isn't in the docker group yet in this session.
  if docker info >/dev/null 2>&1; then
    docker compose build api
  else
    warn "docker group not active in this session; falling back to sudo."
    sudo docker compose build api
  fi
}

dc() {
  # docker-compose wrapper that sudos transparently when the group hasn't propagated yet.
  if docker info >/dev/null 2>&1; then
    docker compose "$@"
  else
    sudo docker compose "$@"
  fi
}

# ────────────────────────────────────────────────────────────────────────────
# 4. VAPID key pair (skip if private key already exists)
# ────────────────────────────────────────────────────────────────────────────
generate_vapid() {
  if [ -f backend/secrets/vapid_private.pem ]; then
    log "VAPID private key already exists — leaving as-is."
    return
  fi
  log "Generating VAPID key pair…"
  mkdir -p backend/secrets
  # Capture the printed public key. The script logs to stderr+stdout; we want
  # the line after "VAPID public key" — grep it out and patch .env.
  OUTPUT=$(dc run --rm api python scripts/generate_vapid_keys.py 2>&1)
  echo "$OUTPUT"
  PUB_KEY=$(echo "$OUTPUT" | awk '/VAPID public key/ {found=1; next} found && NF {print $NF; exit}')
  if [ -z "$PUB_KEY" ]; then
    die "Could not extract VAPID public key from generate_vapid_keys.py output."
  fi
  log "Patching backend/.env with VAPID_PUBLIC_KEY=${PUB_KEY:0:12}…"
  if grep -q '^VAPID_PUBLIC_KEY=' backend/.env; then
    sed -i.bak "s|^VAPID_PUBLIC_KEY=.*|VAPID_PUBLIC_KEY=${PUB_KEY}|" backend/.env
  else
    echo "VAPID_PUBLIC_KEY=${PUB_KEY}" >> backend/.env
  fi
}

# ────────────────────────────────────────────────────────────────────────────
# 4b. SFTP password (skip if .env already has one)
# ────────────────────────────────────────────────────────────────────────────
generate_sftp_password() {
  if grep -qE '^SFTP_PASSWORD=.{8,}' backend/.env 2>/dev/null; then
    log "SFTP_PASSWORD already set — leaving as-is."
    return
  fi
  # Strip slashes and equals from base64 so the password is safe in URLs /
  # config fields. 30 chars of random base64 → ~180 bits of entropy.
  PASS=$(openssl rand -base64 30 | tr -d '/+=' | head -c 30)
  if grep -q '^SFTP_PASSWORD=' backend/.env 2>/dev/null; then
    sed -i.bak "s|^SFTP_PASSWORD=.*|SFTP_PASSWORD=${PASS}|" backend/.env
  else
    echo "SFTP_PASSWORD=${PASS}" >> backend/.env
  fi
  log "Generated SFTP password (write into Reolink: backend/.env SFTP_PASSWORD)"
}

# ────────────────────────────────────────────────────────────────────────────
# 4c. Open UFW port 22222 for the SFTP container
# ────────────────────────────────────────────────────────────────────────────
open_sftp_port() {
  if sudo ufw status 2>/dev/null | grep -q '22222/tcp'; then
    log "UFW already allows 22222/tcp — leaving as-is."
    return
  fi
  log "Opening UFW 22222/tcp for SFTP (Reolink → camera uploads)…"
  sudo ufw allow 22222/tcp comment 'SFTP for Reolink snapshots'
}

# ────────────────────────────────────────────────────────────────────────────
# 5. Frontend build
# ────────────────────────────────────────────────────────────────────────────
build_frontend() {
  if [ -d frontend/dist ] && [ "$(find frontend/dist -newer frontend/src -type f 2>/dev/null | head -1)" ]; then
    log "Frontend already built and newer than src — skipping."
    return
  fi
  log "Building frontend (npm ci + vite build)…"
  ( cd frontend && npm ci --no-audit --no-fund && npm run build )
}

# ────────────────────────────────────────────────────────────────────────────
# 6. Up and verify
# ────────────────────────────────────────────────────────────────────────────
start_stack() {
  log "Bringing the stack up (api + caddy)…"
  dc up -d
  sleep 3
  log "Container status:"
  dc ps
}

verify_tls() {
  log "Waiting up to 60s for Caddy to issue Let's Encrypt cert + serve /api/health…"
  for i in $(seq 1 30); do
    if curl -fsS -m 5 https://birdwatcher.ryanhoulette.com/api/health >/dev/null 2>&1; then
      log "✓ https://birdwatcher.ryanhoulette.com/api/health is responding"
      return 0
    fi
    sleep 2
  done
  warn "Health check didn't respond within 60s. Tail Caddy logs to debug:"
  warn "  docker compose logs caddy --tail 30"
  return 1
}

# ────────────────────────────────────────────────────────────────────────────
# 7. Yard calibration (optional — only if Haikubox creds are set + file missing)
# ────────────────────────────────────────────────────────────────────────────
calibrate() {
  if [ -f backend/data/calibration/yard_priors.json ]; then
    log "yard_priors.json already exists — skipping calibration. Re-run with"
    log "  $0 --recalibrate  to force a refresh."
    return
  fi
  if ! grep -qE '^HAIKUBOX_API_KEY=.{10,}' backend/.env || ! grep -qE '^HAIKUBOX_SERIAL=.{6,}' backend/.env; then
    warn "HAIKUBOX_API_KEY or HAIKUBOX_SERIAL not set in .env — skipping calibration."
    return
  fi
  log "Running yard calibration (200 days, ~40s)…"
  dc run --rm api python scripts/calibrate_from_haikubox.py --days 200 || warn "Calibration failed; continuing without yard_priors.json."
}

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
require_env
install_docker
install_node
build_api
generate_vapid
generate_sftp_password
open_sftp_port
build_frontend
start_stack
verify_tls
calibrate
log "Done. Manual remaining steps (DEPLOY.md):"
log "  10. Configure Reolink camera to HTTP-push to https://birdwatcher.ryanhoulette.com/api/ingest/motion"
log "  11. End-to-end smoke test (wave hand at camera)"
log "  12. Enable push notifications in the PWA on your wife's Android phone"
