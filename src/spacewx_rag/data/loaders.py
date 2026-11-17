"""Readers for the raw sources (notebooks 01 and 02). Missing values become NaN."""

from __future__ import annotations

import gzip
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from spacewx_rag.utils.config import Paths

MAG_KEEP = [
    "time",
    "sample_count",
    "measurement_mode",
    "measurement_range",
    "bt",
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "backfill_flag",
    "future_packet_time_flag",
    "old_packet_time_flag",
    "fill_flag",
    "possible_saturation_flag",
    "calibration_mode_flag",
    "maneuver_flag",
    "low_sample_count_flag",
    "overall_quality",
]
FC_KEEP = [
    "time",
    "sample_count",
    "proton_speed",
    "proton_density",
    "proton_temperature",
    "proton_vx_gsm",
    "proton_vy_gsm",
    "proton_vz_gsm",
    "alpha_density",
    "backfill_flag",
    "old_packet_time_flag",
    "fill_flag",
    "normal_mode_flag",
    "proton_peak_unresolved_flag",
    "proton_alpha_confusion_flag",
    "proton_peak_unbounded_flag",
    "proton_peak_loss_flag",
    "large_flow_angle_flag",
    "possible_second_proton_peak_flag",
    "calibration_mode_flag",
    "maneuver_flag",
    "reduced_proton_quality_flag",
    "bad_proton_speed_flag",
    "bad_proton_density_flag",
    "bad_proton_temperature_flag",
    "low_proton_velocity_sample_count_flag",
    "overall_quality",
]

OMNI_COLS = {
    0: "year",
    1: "doy",
    2: "hour",
    8: "B",
    16: "Bz_gsm",
    22: "T",
    23: "N",
    24: "V",
    38: "Kp",
    40: "Dst",
    49: "ap",
    50: "f107",
}
OMNI_FILLS = {
    "B": 999.9,
    "Bz_gsm": 999.9,
    "T": 9999999.0,
    "N": 999.9,
    "V": 9999.0,
    "Kp": 99,
    "Dst": 99999,
    "ap": 999,
    "f107": 999.9,
}


def read_nc_gz(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one gzipped DSCOVR NetCDF file into a DataFrame (time in UTC)."""
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.nc"
        with gzip.open(path, "rb") as a, open(out, "wb") as b:
            shutil.copyfileobj(a, b)
        with xr.open_dataset(out, decode_times=False) as ds:
            df = ds.to_dataframe().reset_index()
            attrs = dict(ds.attrs)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df, attrs


def latest_versions(files: list[Path]) -> list[Path]:
    """Several days have reprocessed files: keep the latest processing stamp per day."""
    tbl = pd.DataFrame({"path": files})
    tbl["day"] = tbl["path"].map(lambda p: re.search(r"_s(\d{8})", p.name).group(1))
    tbl["proc"] = tbl["path"].map(lambda p: re.search(r"_p(\d{14})", p.name).group(1))
    return tbl.sort_values("proc").groupby("day").tail(1).sort_values("day")["path"].tolist()


def build_interim(paths: Paths | None = None, log=print) -> None:
    """Concatenate every DSCOVR magnetometer (m1m) and Faraday cup (f1m) file into data/interim/."""
    from tqdm.auto import tqdm

    p = paths or Paths()
    p.interim.mkdir(parents=True, exist_ok=True)
    for kind, keep, floats, out in [
        ("m1m", MAG_KEEP, ["bt", "bx_gsm", "by_gsm", "bz_gsm"], "dscovr_mag_1min_all.parquet"),
        (
            "f1m",
            FC_KEEP,
            [
                "proton_speed",
                "proton_density",
                "proton_temperature",
                "proton_vx_gsm",
                "proton_vy_gsm",
                "proton_vz_gsm",
                "alpha_density",
            ],
            "dscovr_fc_1min_all.parquet",
        ),
    ]:
        files = latest_versions(sorted((p.raw / "dscovr" / kind).glob("*/*/*.nc.gz")))
        parts, failed = [], []
        for f in tqdm(files, desc=kind):
            try:
                parts.append(
                    read_nc_gz(f)[0].reindex(columns=keep)
                )  # older f1m files lack some flags
            except Exception as e:  # noqa: BLE001 - report and continue
                failed.append((f.name, str(e)))
        df = pd.concat(parts, ignore_index=True)
        for c in floats:
            df[c] = df[c].astype("float32")
        df.to_parquet(p.interim / out)
        log(f"{kind}: {len(files)} files, {len(failed)} failed, {df.shape} -> {out}")


def read_omni(paths: Paths, idx: pd.DatetimeIndex) -> pd.DataFrame:
    files = sorted((paths.raw / "dst").glob("omni2_hourly_20*.dat"))
    om = pd.concat(
        [
            pd.read_csv(
                f, sep=r"\s+", header=None, usecols=list(OMNI_COLS), names=list(OMNI_COLS.values())
            )
            for f in files
        ],
        ignore_index=True,
    )
    om.columns = [OMNI_COLS[k] for k in sorted(OMNI_COLS)]
    om["time"] = (
        pd.to_datetime(om["year"].astype(str), format="%Y")
        + pd.to_timedelta(om["doy"] - 1, unit="D")
        + pd.to_timedelta(om["hour"], unit="h")
    )
    for c, fv in OMNI_FILLS.items():
        om[c] = om[c].where(om[c] != fv, np.nan)
    return om.set_index("time").drop(columns=["year", "doy", "hour"]).reindex(idx)


def read_kyoto_dst(paths: Paths, idx: pd.DatetimeIndex) -> pd.DataFrame:
    files = sorted((paths.raw / "dst").glob("kyoto_hour_dst_20*.csv"))
    ky = pd.concat(
        [
            pd.read_csv(f, header=None, names=["time", "dst", "dst_version"], parse_dates=["time"])
            for f in files
        ],
        ignore_index=True,
    )
    ky["hour"] = (
        ky["time"].dt.tz_localize(None).dt.floor("h")
    )  # stamps sit mid-hour; use the hour start
    return ky.set_index("hour")[["dst", "dst_version"]].reindex(idx)


def read_gfz_kp(paths: Paths, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Kp and ap held over each 3-hour slot, plus the day's definitive flag D."""
    gcols = (
        ["year", "month", "day"]
        + [f"Kp{i}" for i in range(1, 9)]
        + [f"ap{i}" for i in range(1, 9)]
        + ["D"]
    )
    gfz = pd.read_csv(
        paths.raw / "kp" / "gfz_Kp_ap_Ap_SN_F107_since_1932.txt",
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2] + list(range(7, 23)) + [27],
        names=gcols,
    )
    gfz["date"] = pd.to_datetime(gfz[["year", "month", "day"]])
    g = gfz.set_index("date")
    days, slot, rows = idx.normalize(), idx.hour // 3, np.arange(len(idx))
    return pd.DataFrame(
        {
            "kp": g[[f"Kp{i}" for i in range(1, 9)]].reindex(days).to_numpy()[rows, slot],
            "ap": g[[f"ap{i}" for i in range(1, 9)]].reindex(days).to_numpy()[rows, slot],
            "kp_D": g["D"].reindex(days).to_numpy(),
        },
        index=idx,
    )


def read_celestrak_f107(paths: Paths) -> pd.DataFrame:
    ct = pd.read_csv(
        paths.raw / "f107" / "celestrak_SW-All.csv",
        parse_dates=["DATE"],
        usecols=["DATE", "F10.7_OBS", "F10.7_ADJ", "F10.7_DATA_TYPE"],
    ).set_index("DATE")
    return ct[ct["F10.7_DATA_TYPE"].isin(["OBS", "INT"])]  # drop PRD / PRM predictions
