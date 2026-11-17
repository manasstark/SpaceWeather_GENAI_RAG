"""Fetch the real-time NOAA SWPC feeds (no key needed) and keep a local snapshot.

Each fetch is saved under data/live/snapshots/, so the app can fall back to the last good
snapshot when the network or NOAA is unavailable. Only the active L1 monitor's rows and the
columns the pipeline uses are stored, which keeps a snapshot small (about 200 KB).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from spacewx_rag.utils.config import Paths, load_params

WIND_COLS = ["time_tag", "source", "proton_speed", "proton_density", "proton_temperature"]
MAG_COLS = ["time_tag", "source", "bt", "bx_gsm", "by_gsm", "bz_gsm"]


def live_dir(paths: Paths | None = None) -> Path:
    return (paths or Paths()).root / "data" / "live"


def _active(rows: list[dict], cols: list[str]) -> list[dict]:
    """Rows of the monitor NOAA flags as active, trimmed to the columns we use."""
    return [{c: r.get(c) for c in cols} for r in rows if r.get("active")]


def fetch_feeds(session: requests.Session | None = None) -> dict:
    """Download every feed. Raises requests.RequestException on any failure."""
    cfg = load_params()["live"]
    s = session or requests.Session()
    raw = {}
    for name, path in cfg["feeds"].items():
        r = s.get(f"{cfg['base_url']}/{path}", timeout=cfg["timeout_s"])
        r.raise_for_status()
        raw[name] = r.json()
    return {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": cfg["base_url"],
        "wind": _active(raw["wind"], WIND_COLS),
        "mag": _active(raw["mag"], MAG_COLS),
        "kp": raw["kp"],
        "dst": raw["dst"],
        "f107": [
            {k: r.get(k) for k in ("time_tag", "flux", "reporting_schedule")} for r in raw["f107"]
        ],
    }


def save_snapshot(snap: dict, paths: Paths | None = None) -> Path:
    d = live_dir(paths) / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    stamp = snap["fetched_at"].replace(":", "").replace("-", "")
    path = d / f"snapshot_{stamp}.json"
    path.write_text(json.dumps(snap))
    return path


def latest_snapshot(paths: Paths | None = None) -> dict | None:
    """The newest saved snapshot, else the sample shipped with the repository."""
    d = live_dir(paths)
    files = sorted((d / "snapshots").glob("snapshot_*.json"))
    path = files[-1] if files else d / "sample_snapshot.json"
    return json.loads(path.read_text()) if path.exists() else None


def get_live_data(paths: Paths | None = None) -> tuple[dict, str]:
    """(snapshot, status). Fetch fresh data; on failure fall back to the last saved snapshot."""
    try:
        snap = fetch_feeds()
        save_snapshot(snap, paths)
        return snap, "fresh"
    except (requests.RequestException, ValueError) as e:
        snap = latest_snapshot(paths)
        if snap is None:
            raise RuntimeError(f"NOAA feeds unavailable and no saved snapshot: {e}") from e
        return (
            snap,
            f"offline: using the snapshot fetched at {snap['fetched_at']} ({type(e).__name__})",
        )
