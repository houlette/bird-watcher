# BirdWatcher

AI-powered bird-feeder camera that complements a Haikubox (audio bird-ID) with vision-based detection and species classification.

- **Camera**: Reolink RLC-811WA (4K, WiFi, varifocal) → motion-event clips pushed via HTTP
- **Backend**: FastAPI on a small cloud VM. YOLO11-nano detection → per-bird IoU tracking → EfficientNetV2 (NABirds) species classifier → Bayesian fusion with Haikubox audio prior + seasonal prior → multi-frame voting
- **Frontend**: Vite/React/TS PWA with Web Push for rare-species notifications
- **Host**: `birdwatcher.ryanhoulette.com`

See `/Users/ryan/.claude/plans/my-wife-has-many-buzzing-noodle.md` for the full design.

## Development

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Deployment

```bash
docker compose up -d
```
