"""The hourly aligned table (notebook 02): DSCOVR shifted to Earth arrival, OMNI fallback, indices.

Output: data/processed/aligned_hourly.parquet, 87,024 rows x 20 columns (decision 21).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacewx_rag.data.loaders import read_celestrak_f107, read_gfz_kp, read_kyoto_dst, read_omni
from spacewx_rag.utils.config import Paths, load_params


def hour_index() -> pd.DatetimeIndex:
    w = load_params()["analysis_window"]
    return pd.date_range(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]), freq="h")


def clean_magnetometer(mag: pd.DataFrame) -> pd.DataFrame:
    """Blank maneuver and saturation minutes (decision 11); minute data indexed by time."""
    m = mag.copy()
    bad = np.zeros(len(m), dtype=bool)
    for flag in load_params()["alignment"]["magnetometer_blank_flags"]:
        bad |= (m[flag] == 1).to_numpy()
    m.loc[bad, ["bt", "bx_gsm", "by_gsm", "bz_gsm"]] = np.nan
    return m.set_index("time")


def hourly_plasma_l1(fc: pd.DataFrame, idx: pd.DatetimeIndex, min_minutes: int) -> pd.DataFrame:
    """Unshifted hourly DSCOVR plasma and the share of reduced-quality minutes (used to pick the source)."""
    f = fc.set_index("time")
    hp = f[["proton_speed", "proton_density", "proton_temperature"]].resample("h").mean()
    hp["n_plasma"] = f["proton_speed"].resample("h").count()
    hp["flag_share"] = f["reduced_proton_quality_flag"].resample("h").mean()
    hp = hp.reindex(idx)
    hp["n_plasma"] = hp["n_plasma"].fillna(0).astype(int)
    hp.loc[
        hp["n_plasma"] < min_minutes, ["proton_speed", "proton_density", "proton_temperature"]
    ] = np.nan
    return hp.rename(
        columns={
            "proton_speed": "v_dscovr",
            "proton_density": "n_dscovr",
            "proton_temperature": "t_dscovr",
        }
    )


def f107_columns(ct: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Adjusted and observed F10.7 per hour, with interpolated-day and spike flags (decision 20)."""
    cfg = load_params()["alignment"]["f107_spike"]
    days = idx.normalize()
    d = ct.reindex(days)
    adj = ct.loc[idx[0] : idx[-1], "F10.7_ADJ"]
    med = adj.rolling(
        cfg["median_window_days"], center=True, min_periods=cfg["min_periods"]
    ).median()
    spike = adj > cfg["factor"] * med
    return pd.DataFrame(
        {
            "f107_adj": d["F10.7_ADJ"].to_numpy(),
            "f107_obs": d["F10.7_OBS"].to_numpy(),
            "f107_interp": (d["F10.7_DATA_TYPE"] == "INT").to_numpy(),
            "f107_spike": spike.reindex(days).to_numpy(),
            "f107_med7": med.reindex(days).to_numpy(),
        },
        index=idx,
    )


def to_arrival(df: pd.DataFrame, speed: pd.Series, distance_km: float) -> pd.DataFrame:
    """Shift each minute by its L1-to-Earth delay: distance / speed of that hour (decision 19)."""
    s = df.copy()
    v = speed.reindex(s.index.floor("h")).to_numpy()
    s.index = s.index + pd.to_timedelta(distance_km / v, unit="s")
    return s


def build_aligned(paths: Paths | None = None, save: bool = False) -> pd.DataFrame:
    p = paths or Paths()
    cfg = load_params()["alignment"]
    idx = hour_index()
    nmin = cfg["min_valid_minutes_per_hour"]

    mag = pd.read_parquet(p.interim / "dscovr_mag_1min_all.parquet")
    fc = pd.read_parquet(p.interim / "dscovr_fc_1min_all.parquet")
    m = clean_magnetometer(mag)
    hp = hourly_plasma_l1(fc, idx, nmin)
    om = read_omni(p, idx)

    # plasma source per hour (decision 12) and the speed used for the delay
    plasma_src = np.where(
        hp["v_dscovr"].notna() & (hp["flag_share"] < cfg["plasma_fallback_flag_share"]),
        "dscovr",
        np.where(om["V"].notna(), "omni", "missing"),
    )
    vf = pd.Series(np.where(plasma_src == "dscovr", hp["v_dscovr"], om["V"]), index=idx).fillna(
        cfg["speed_when_missing_km_s"]
    )
    D = float(cfg["delay_distance_km"])

    # field: DSCOVR shifted to arrival time, hourly means
    fld = m[["bx_gsm", "by_gsm", "bz_gsm", "bt"]].dropna(subset=["bz_gsm"])
    s = to_arrival(fld, vf, D)
    g = s.groupby(s.index.floor("h"))
    fe = g.mean()
    fe["bz_min"] = g["bz_gsm"].min()
    fe = fe[g["bz_gsm"].count() >= nmin].reindex(idx)

    # plasma: mask unreliable L1 hours first, then shift
    mf = fc.set_index("time")[["proton_speed", "proton_density", "proton_temperature"]]
    reliable = (
        pd.Series(plasma_src == "dscovr", index=idx)
        .reindex(mf.index.floor("h"))
        .fillna(False)
        .to_numpy()
        .astype(bool)
    )
    mf = mf[reliable].dropna(subset=["proton_speed"])
    pz = to_arrival(mf, vf, D)
    g2 = pz.groupby(pz.index.floor("h"))
    pe = g2.mean()
    pe = pe[g2["proton_speed"].count() >= nmin].reindex(idx)

    ky, kp, f107 = (
        read_kyoto_dst(p, idx),
        read_gfz_kp(p, idx),
        f107_columns(read_celestrak_f107(p), idx),
    )

    a = pd.DataFrame(index=idx)
    a["dst"], a["dst_version"] = ky["dst"], ky["dst_version"]
    a["kp"], a["ap"], a["kp_D"] = kp["kp"], kp["ap"], kp["kp_D"]
    for c in ["f107_adj", "f107_obs", "f107_interp", "f107_spike", "f107_med7"]:
        a[c] = f107[c]
    a["bz"] = fe["bz_gsm"].fillna(om["Bz_gsm"])
    a["bt"] = fe["bt"].fillna(om["B"])
    a["bz_src"] = np.where(
        fe["bz_gsm"].notna(), "dscovr", np.where(om["Bz_gsm"].notna(), "omni", "missing")
    )
    a["bx"], a["by"], a["bz_min"] = fe["bx_gsm"], fe["by_gsm"], fe["bz_min"]
    a["v"] = pe["proton_speed"].fillna(om["V"])
    a["n"] = pe["proton_density"].fillna(om["N"])
    a["t"] = pe["proton_temperature"].fillna(om["T"])
    a["plasma_src"] = np.where(
        pe["proton_speed"].notna(), "dscovr", np.where(om["V"].notna(), "omni", "missing")
    )
    a.index.name = "time_utc"
    if save:
        p.processed.mkdir(parents=True, exist_ok=True)
        a.to_parquet(p.aligned)
    return a
