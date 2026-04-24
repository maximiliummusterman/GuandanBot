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
- `api/health.py`
- `bot_api.py`
- `guandan_arena.py`
- `requirements.txt`
- `.python-version`
- `vercel.json`
- `README.md`
- `run_bot_api.ps1`
- one latest checkpoint in `checkpoints/`

Remove old checkpoints, test files, and frontend reference files from the deploy repo if you do not need them.

## Local Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bot_api.ps1 -Port 8765
```

Optional explicit checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bot_api.ps1 -Checkpoint .\checkpoints\guandan_latest.pt
```

## Vercel Deploy

1. Put the latest model file in `checkpoints/`.
2. Push this repo to GitHub.
3. Import the repo into Vercel.
4. Leave the framework as auto-detected Python.
5. If your checkpoint is not the latest `.pt` file in `checkpoints/`, set `GUANDAN_CHECKPOINT` in Vercel project environment variables.
6. Deploy.

Notes:

- `api/decision.py` and `api/health.py` are the Vercel entrypoints and re-export the WSGI app from `bot_api.py`.
- `.python-version` pins Python `3.12`.
- `vercel.json` excludes local/dev files from the Python bundle.
- Keep only the latest checkpoint in the deploy repo to avoid unnecessary bundle size.

## Endpoints

### `GET /api/health`

Returns basic runtime info:

```json
{
  "status": "ok",
  "device": "cpu",
  "checkpoint": "checkpoints/guandan_ep256000.pt"
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
- For production deploys, keep only the latest checkpoint in `checkpoints/`.

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
