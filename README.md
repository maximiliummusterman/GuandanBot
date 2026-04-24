# Guandan Bot API

Thin Python API for the Guandan website bots.

This repo is set up for a normal Python web-service deployment on Render.

It exposes:

- `POST /decision`
- `GET /health`

The API accepts website-style match state and returns one of:

- `action`
- `tributeCard`
- `returnCard`
- `skipTribute`

## Why Render

This API uses PyTorch for inference.
That is a poor fit for Vercel serverless bundle limits, but a normal Python web service on Render handles it well.

## Files You Need

For a clean deploy repo, keep:

- `bot_api.py`
- `guandan_arena.py`
- `requirements.txt`
- `.python-version`
- `Procfile`
- `render.yaml`
- `README.md`
- `run_bot_api.ps1`
- `checkpoint.pt`

You do not need old checkpoints, test files, or frontend reference files in the deploy repo.

## Local Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bot_api.ps1 -Port 8765 -Checkpoint .\checkpoint.pt
```

If you omit `-Checkpoint`, the API tries:

1. `GUANDAN_CHECKPOINT`
2. `checkpoint.pt`
3. `checkpoints/latest.pt`
4. latest `.pt` file in `checkpoints/`

## Render Deploy

### Option 1: Blueprint

This repo includes [render.yaml](</c:/Users/Oscar Lin/OneDrive/Dokumente/Guandan AI/render.yaml:1>).
It is configured for Render's `free` web-service plan.

Steps:

1. Put your latest model at repo root as `checkpoint.pt`.
2. Push the repo to GitHub.
3. In Render, choose `New +` -> `Blueprint`.
4. Select this repo.
5. Deploy.

Notes:

- If Render asked for a card before, it was because the Blueprint was set to the paid `starter` plan.
- Free web services can spin down after idle time, so the first bot request after a pause can be slow.

### Option 2: Manual Web Service

If you prefer configuring Render manually, use:

- Runtime: `Python`
- Instance Type: `Free`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 bot_api:app`
- Health Check Path: `/health`

Environment variables:

- `PYTHON_VERSION=3.12.9`
- `GUANDAN_CHECKPOINT=checkpoint.pt`

## Endpoints

### `GET /health`

Returns basic runtime info:

```json
{
  "status": "ok",
  "device": "cpu",
  "checkpoint": "checkpoint.pt"
}
```

### `POST /decision`

Minimal action request:

```json
{
  "requestType": "action",
  "match": {
    "gameStatus": "playing",
    "currentSeat": "1",
    "currentRoundLevelRank": "Blue",
    "levelRankBlue": "2",
    "levelRankRed": "2",
    "trickLastPlay": [],
    "lastTrickSeat": null,
    "roundNumber": 1
  },
  "players": [
    {
      "id": "p1",
      "user_id": "u1",
      "seat": "1",
      "team": "Blue",
      "hand": [],
      "finishPlace": null
    },
    {
      "id": "p2",
      "user_id": "u2",
      "seat": "2",
      "team": "Red",
      "hand": [],
      "finishPlace": null
    },
    {
      "id": "p3",
      "user_id": "u3",
      "seat": "3",
      "team": "Blue",
      "hand": [],
      "finishPlace": null
    },
    {
      "id": "p4",
      "user_id": "u4",
      "seat": "4",
      "team": "Red",
      "hand": [],
      "finishPlace": null
    }
  ]
}
```

Accepted aliases:

- `match` or `currentMatch`
- `players`, `matchPlayers`, or `allMatchPlayers`
- `currentPlayer` or `seat`

Important:

- `players` must contain all four seats.
- The acting player's `hand` must match the current website state.
- Put only one deploy-time model in the repo: `checkpoint.pt`.

Possible response fields:

- `decisionType`
- `seat`
- `action`
- `actionIdx`
- `pass`
- `tributeCard`
- `returnCard`
- `tributeState`
- `skipTribute`
- `checkpoint`

## Frontend Wiring

Once deployed, your website should call:

- `https://YOUR-RENDER-SERVICE.onrender.com/health`
- `https://YOUR-RENDER-SERVICE.onrender.com/decision`

If your frontend is still hosted on Vercel, you can either call Render directly or add a Vercel rewrite that proxies `/api/bot/*` to your Render service.
