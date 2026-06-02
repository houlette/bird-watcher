# BirdWatcher project conventions

## Default post-edit workflow

After **any** code change to the BirdWatcher repo, run all three steps
without being asked. The user wants this to be the default:

1. **Deploy to the VM**:
   ```
   ./scripts/deploy_to_server.sh ryan@birdwatcher.ryanhoulette.com
   ```
   (script lives at `scripts/deploy_to_server.sh`, not repo root)
2. **Commit** on `main` with a Claude co-author line.
3. **Push** to `origin/main`.

The user works directly on `main` — no feature branches. Don't open PRs
unless asked.

Skip the deploy step only for changes that obviously can't affect the
running system (e.g., edits to `CLAUDE.md`, `.claude/`, throwaway
experiment scripts under `backend/scripts/sweep/` that are gitignored
output, or `*.md` docs that don't ship in the image).

## Untracked paths to leave alone

These appear in `git status` but are NOT meant to be committed:
- `backend/data/` — runtime data (DB, clips, calibration, heatmaps)
- `backend/scripts/sweep/run_*.py` — local experiment scripts
- `backend/tests/test_fuse_crops.py` — local scratch test

Stage files by name rather than `git add -A` so these stay out.

## Deploy target

- Host: `ryan@birdwatcher.ryanhoulette.com`
- Health check: `https://birdwatcher.ryanhoulette.com/api/health`
- The deploy script handles rsync + remote bootstrap + container
  rebuild + Caddy cert verification. Treat its tail output as the
  success signal.
