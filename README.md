# S&P 500 Breadth Aggregators

A Macro Ops-style **Breadth Components** score and **Breadth Thrust Aggregator** for the S&P 500, rebuilt bottom-up from point-in-time index constituents and refreshed automatically each US trading day.

**Live dashboard:** `https://<your-username>.github.io/<repo-name>/`

## What's in the repo

| Path | Purpose |
|---|---|
| `rebuild.py` | The whole pipeline in one file: membership, prices, breadth series, signals, evidence tables and the HTML template. Writes `docs/index.html`. |
| `requirements.txt` | Pinned Python dependencies. |
| `.github/workflows/refresh.yml` | Scheduled GitHub Action that rebuilds the page and commits it. |
| `docs/index.html` | The published dashboard, served by GitHub Pages. |
| `docs/.nojekyll` | Tells Pages to serve the file as-is. |

## How the refresh works

1. The workflow runs at 22:17 UTC Monday to Friday, about two hours after the US close (09:17 Melbourne in AEDT).
2. It restores the price cache from the previous run, so it only downloads the last three weeks of prices (about 2 minutes). The first run, or a manual run with **refresh** ticked, downloads full history from 1998 (about 6 minutes).
3. `rebuild.py` regenerates `docs/index.html` and the workflow commits it. GitHub Pages republishes within a minute or two.
4. **Sanity gate:** if fewer than 90% of current members are priced, or the data ends earlier than the page already published, the build stops without writing. The live page keeps its last good version, the run shows red, and GitHub emails you.

## Run locally

```bash
pip install -r requirements.txt
python rebuild.py                 # writes docs/index.html
python rebuild.py --out my.html   # or anywhere else
python rebuild.py --refresh       # ignore the cache, re-download everything
```

## Data sources

- Prices: Yahoo Finance via `yfinance` (daily closes; ^GSPC for the index).
- Membership: [fja05680/sp500](https://github.com/fja05680/sp500) point-in-time history, rolled forward with Wikipedia's current constituent list.

Method, thresholds and caveats are documented in the dashboard's "Method and caveats" panel.
