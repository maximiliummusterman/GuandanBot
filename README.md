# Guandan Bot API

Thin Python API for the Guandan website bots.

It exposes:

- `POST /api/decision`
- `GET /api/health`

The API accepts website-style match state and returns one of:

- `action`
- `tributeCard`
- `returnCard`
- `skipTribute`

## Repo Layout

Keep the deploy repo small. In practice it should contain:

- `api/decision.py`
- `bot_api.py`
- `checkpoint.pt`
- `guandan_arena.py`
- `requirements.txt`
- `.python-version`
- `vercel.json`
- `README.md`
- `run_bot_api.ps1`

For Vercel, prefer a single root-level `checkpoint.pt`.
The `checkpoints/` folder is excluded from the Vercel function bundle.

## Local Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bot_api.ps1 -Port 8765
```

Optional explicit checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bot_api.ps1 -Checkpoint .\checkpoints\guandan_latest.pt
```

## Vercel Deploy

1. Put the latest model file at repo root as `checkpoint.pt`.
2. Push this repo to GitHub.
3. Import the repo into Vercel.
4. Leave the framework as auto-detected Python.
5. If you want a different checkpoint location, set `GUANDAN_CHECKPOINT` in Vercel project environment variables.
6. Deploy.

Notes:

- `api/decision.py` is the only Vercel function entrypoint.
- `GET /api/health` is rewritten to that same function to avoid bundling PyTorch twice.
- `.python-version` pins Python `3.12`.
- `requirements.txt` uses CPU-only PyTorch wheels to avoid the oversized default Linux package.
- `vercel.json` excludes local/dev files and the `checkpoints/` training folder from the Python bundle.
- If Vercel still exceeds the function size limit after this, the remaining bottleneck is PyTorch itself and you will likely need a non-Lambda host.

## Endpoints

### `GET /api/health`

Returns basic runtime info:

```json
{
  "status": "ok",
  "device": "cpu",
  "checkpoint": "checkpoint.pt"
}
```

### `POST /api/decision`

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
- For Vercel deploys, use a single root `checkpoint.pt`.

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
