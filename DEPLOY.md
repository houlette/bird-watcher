# Deploying BirdWatcher

End-to-end guide for standing up the cloud half of BirdWatcher on a fresh Ubuntu VM. Assumes you've already:

- Bought the **Reolink RLC-811WA** camera (arrives separately; configured at the end of this doc)
- Run `scripts/calibrate_from_haikubox.py` locally and committed/copied the resulting `yard_priors.json` (optional but recommended — see [Yard calibration](#yard-calibration))
- Own the domain `ryanhoulette.com` and can edit its DNS

The whole deploy takes about 30 minutes of clock time, most of it waiting for ML deps to install in the container.

## 1. Provision the VM

Pick one. All three are fine for this workload.

| Provider / plan | Cost | Specs | Why pick it |
|---|---|---|---|
| Hetzner **CAX21** (ARM shared) | ~$6/mo | 4 vCPU, 8 GB RAM, 80 GB SSD | **Recommended.** All our deps have ARM64 wheels; we're already running ARM Docker images during dev. Best price/perf for bursty inference. |
| Contabo **Cloud VPS S** | ~$7/mo | 4 vCPU, 8 GB RAM, 200 GB NVMe | If you want more disk for clip retention; US-based. |
| Hetzner **CPX31** (x86 shared) | ~$13/mo | 4 vCPU, 8 GB RAM, 80 GB SSD | x86 if you ever need a closed-source binary that doesn't ship ARM. |

Notes on what to skip:

- **Hetzner CCX (dedicated)**: noisy-neighbor protection isn't worth the 2.5× price for a bird camera that does ~20 minutes of CPU/day. Easy to upgrade to later if you ever see contention.
- **AWS Spot**: termination = data loss.
- **AWS Lambda**: wrong shape (cold starts + always-on polling).

Whichever you choose:

- **OS**: Ubuntu 24.04 LTS
- **SSH**: paste in your public key during creation; password auth off
- **Firewall**: allow 22 (SSH), 80 (HTTP), 443 (HTTPS) only
- Note the public IPv4 address — you'll need it in the next step

## 2. Point DNS

In your domain registrar, add an A record:

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `birdwatcher` | `<VM_PUBLIC_IPV4>` | 300 |

Wait until `dig +short birdwatcher.ryanhoulette.com` from your laptop returns the VM IP. Usually < 2 minutes; can take longer if your registrar caches aggressively.

## 3. Initial server hardening

SSH in as root (or whatever user the provider configured), then:

```bash
# Create a non-root user so we don't run docker as root
adduser --disabled-password --gecos "" ryan
usermod -aG sudo ryan
mkdir -p /home/ryan/.ssh
cp /root/.ssh/authorized_keys /home/ryan/.ssh/
chown -R ryan:ryan /home/ryan/.ssh
chmod 700 /home/ryan/.ssh
chmod 600 /home/ryan/.ssh/authorized_keys

# Allow ryan to sudo without a password (because --disabled-password above
# means there IS no password to type, and bootstrap_server.sh runs sudo
# non-interactively). Safe given SSH-key-only auth: anyone with sudo here
# already has SSH access.
echo 'ryan ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/ryan
chmod 0440 /etc/sudoers.d/ryan

# Lock down SSH
sed -i 's/^#*PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl reload ssh

# Auto-install security updates
apt update && apt install -y unattended-upgrades ufw
dpkg-reconfigure --priority=low unattended-upgrades

# Firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
```

Disconnect, then re-SSH as `ryan@birdwatcher.ryanhoulette.com` to confirm it works before continuing.

## 4. Install Docker

```bash
# Official Docker repo install (Ubuntu 24.04)
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Let our user run docker without sudo
sudo usermod -aG docker $USER
newgrp docker  # apply the group change for this session
docker --version  # sanity check
```

## 5. Deploy the code

```bash
git clone https://github.com/<your-account>/BirdWatcher.git  # or whatever the repo URL is
cd BirdWatcher
cp backend/.env.example backend/.env
nano backend/.env
```

Fill in `backend/.env`:

```
HAIKUBOX_API_KEY=weft_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   # from haikubox.com → settings → API keys
HAIKUBOX_SERIAL=XXXXXXXXXXXX                                  # 12 hex chars; emailed when you registered
VAPID_SUBJECT=mailto:you@example.com
# VAPID_PUBLIC_KEY filled in by step 6
```

## 6. Generate VAPID keys

Web Push needs a P-256 key pair. The private key stays on the VM; the public key goes in `.env` for the frontend to fetch.

```bash
docker compose build api   # ~5 min the first time (downloads torch, transformers, ultralytics)
docker compose run --rm api python scripts/generate_vapid_keys.py
```

It writes `backend/secrets/vapid_private.pem` (mode 600, gitignored) and prints the public key. Edit `backend/.env` and paste the public key into `VAPID_PUBLIC_KEY=...`.

## 7. Build the frontend

The Caddy container serves the prebuilt PWA from `frontend/dist`. Build it once:

```bash
# Install node 20 (matches the Vite project's expected version)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

cd frontend
npm ci
npm run build
cd ..
```

You should see `frontend/dist/` with `index.html`, an `assets/` folder, and `sw.js`.

## 8. First boot

```bash
docker compose up -d
docker compose ps   # api + caddy should both be 'running'
docker compose logs caddy --tail=20
```

Caddy auto-issues a Let's Encrypt certificate for `birdwatcher.ryanhoulette.com` on first start — this works only if DNS already points here. Look for a `certificate obtained successfully` line in the logs.

Verify from your laptop:

```bash
curl https://birdwatcher.ryanhoulette.com/api/health
# → {"status": "ok"}
```

Open the URL in a browser. You should see the empty Feed view of the PWA.

## 9. Yard calibration

If you haven't already, run the calibration to build `data/calibration/yard_priors.json`:

```bash
docker compose run --rm api python scripts/calibrate_from_haikubox.py --days 200
```

The pipeline picks up the file live — no restart needed.

## 10. Configure the Reolink RLC-811WA

In the Reolink web client (the camera's IP on your LAN):

1. **Settings → Network → WiFi**: connect to your 5GHz WiFi. Confirm signal strength is ≥ −65 dBm at the mount point.
2. **Settings → Detection → Motion**: enable basic motion detection. **Disable** AI Person/Vehicle/Animal — they're tuned for security cameras and miss small birds.
3. **Settings → Recording → Schedule**: motion-triggered, 24/7.
4. **Settings → Network → HTTP push** (or "Webhook"):
   - URL: `https://birdwatcher.ryanhoulette.com/api/ingest/motion`
   - Method: POST, multipart upload of the motion clip
   - Trigger: on motion event
5. **Live view**: confirm framing. Adjust the varifocal lens until a typical bird at the feeder is ≥ 200 px wide. (At 4K, this is roughly the lens set to 6–10 mm depending on distance.)

## 11. End-to-end smoke test

Wave your hand in front of the camera. Within ~10 seconds:

```bash
docker compose logs api --tail=50 | grep -E "ingest|pipeline"
```

You should see lines like:

```
INFO    POST /api/ingest/motion 200 OK
INFO    pipeline.process: visit 1: 4 tracks
INFO    pipeline.process: visit 1: 2 tracks persisted (after not_a_bird filter)
```

(Your hand will get rejected as not-a-bird, which is correct behavior.)

For a real positive test, point your phone showing a cardinal photo at the camera.

## 12. Enable push notifications

On your wife's Android phone:

1. Visit `https://birdwatcher.ryanhoulette.com` in Chrome.
2. Chrome menu → "Add to Home screen" (installs the PWA).
3. Open the PWA → Settings → "Enable bird notifications".
4. Grant the permission prompt.
5. The next rare-species detection pushes a notification.

## Routine operations

```bash
# Tail logs
docker compose logs -f api
docker compose logs -f caddy

# Restart after a code change
git pull
docker compose build api
cd frontend && npm run build && cd ..
docker compose up -d

# Free disk space — clips and crops older than 30 days
find backend/data/clips -name "*.mp4" -mtime +30 -delete
find backend/data/crops -name "*.jpg" -mtime +30 -delete

# Refresh the yard calibration (every few months — bird populations shift)
docker compose run --rm api python scripts/calibrate_from_haikubox.py --days 200

# Back up the SQLite DB + secrets (do this before any major change)
tar czf birdwatcher-backup-$(date +%F).tar.gz \
  backend/data/birdwatcher.db \
  backend/data/calibration/yard_priors.json \
  backend/secrets/
```

## Troubleshooting

**TLS cert fails to issue.** Caddy log will say `unable to obtain certificate`. Most common cause: DNS not yet propagated, or the firewall blocking port 80. Fix DNS / firewall, then `docker compose restart caddy`.

**Camera uploads succeed but no detections appear.** Tail the API logs; if you see `not_a_bird` rejections, the YOLO/classifier filter is being too aggressive — relax `IN_RANGE_THRESHOLD` in `pipeline/classify.py`. If you see no `pipeline.process` logs at all, the worker isn't picking up new visits — restart the api container.

**Push notifications never arrive.** Check `docker compose logs api | grep -i push`. Common causes: VAPID_PUBLIC_KEY env var missing or doesn't match the key the browser subscribed with (regenerate keys → unsubscribe in PWA Settings → resubscribe).

**Frontend shows 'push not configured'.** Either VAPID_PUBLIC_KEY is empty or the api container can't read it; restart with `docker compose up -d`.
