"""Download the raw space weather data used by notebooks 01-03.

Sources (see data/DATA_SOURCES.md for licences and caveats):
  - GFZ Potsdam      Kp/ap/F10.7 (one text file)
  - CelesTrak        SW-All.csv (Kp, ap, F10.7; one file)
  - Kyoto WDC        hourly Dst via the HAPI service (one CSV per year)
  - NASA OMNI2       hourly yearly files plus the format description
  - NOAA NCEI DSCOVR 1-minute magnetometer (m1m) and Faraday cup (f1m), one gzipped NetCDF per day

Usage (from the repository root):
    python scripts/download_data.py --dry-run          # list what would be downloaded, fetch nothing
    python scripts/download_data.py                    # full download (~390 MB, about an hour)
    python scripts/download_data.py --years 2024       # only one year, for a quick trial (~35 MB)

Files that already exist are skipped, so an interrupted run can simply be started again.
Uses only the Python standard library.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (research; spaceweather capstone)"}
NOAA_ARCHIVE = "https://archive.data.noaa.gov/satellite-spaceweather/"
FIRST_YEAR = 2016                      # DSCOVR became operational in mid 2016


def s3_list(prefix, max_keys=5000):
    """List files under a prefix of the NOAA archive (an S3 bucket behind a proxy)."""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if token:
            q["continuation-token"] = token
        url = NOAA_ARCHIVE + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            xml = r.read().decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]*)", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(url, dest, retries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, \
                    open(str(dest) + ".part", "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            Path(str(dest) + ".part").replace(dest)
            return "ok"
        except Exception as e:                      # network errors: try again
            err = e
    return f"FAIL {url} {err}"


def build_jobs(years):
    today = dt.datetime.now(dt.timezone.utc)
    last_hour = (today - dt.timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    jobs = [
        ("https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt", RAW / "kp" / "gfz_Kp_ap_Ap_SN_F107_since_1932.txt"),
        ("https://celestrak.org/SpaceData/SW-All.csv", RAW / "kp" / "celestrak_SW-All.csv"),
        ("https://celestrak.org/SpaceData/SW-All.csv", RAW / "f107" / "celestrak_SW-All.csv"),
        ("https://wdc.kugi.kyoto-u.ac.jp/hapi/info?id=hour_dst", RAW / "dst" / "kyoto_hour_dst_info.json"),
        ("https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text", RAW / "dst" / "omni2_format_description.txt"),
    ]
    for y in years:
        start = "2016-07-01T00:00:00Z" if y == 2016 else f"{y}-01-01T00:00:00Z"
        end = f"{y + 1}-01-01T00:00:00Z"
        if y == today.year:
            end = last_hour.strftime("%Y-%m-%dT%H:%M:%SZ")
        jobs.append((f"https://wdc.kugi.kyoto-u.ac.jp/hapi/data?id=hour_dst&time.min={start}&time.max={end}&format=csv",
                     RAW / "dst" / f"kyoto_hour_dst_{y}.csv"))
        jobs.append((f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{y}.dat",
                     RAW / "dst" / f"omni2_hourly_{y}.dat"))
        for product, folder in (("MAG/m1m", "m1m"), ("FC/f1m", "f1m")):
            for key in s3_list(f"DSCOVR/DSCOVR/{product}/{y}/"):
                if key.endswith(".nc.gz"):
                    parts = key.split("/")                                  # .../<year>/<month>/<file>
                    jobs.append((NOAA_ARCHIVE + key, RAW / "dscovr" / folder / parts[-3] / parts[-2] / parts[-1]))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded without fetching")
    ap.add_argument("--years", type=int, nargs="+", help="restrict to these years (default: 2016 to now)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    years = args.years or list(range(FIRST_YEAR, dt.date.today().year + 1))
    jobs = build_jobs(years)
    already = sum(1 for _, d in jobs if d.exists() and d.stat().st_size > 0)
    print(f"{len(jobs)} files for years {years[0]}-{years[-1]}; {already} already present; target folder {RAW}")
    if args.dry_run:
        for url, dest in jobs[:5]:
            print("  e.g.", url, "->", dest.relative_to(ROOT))
        return

    failed, done = [], 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda j: fetch(*j), jobs):
            done += 1
            if res.startswith("FAIL"):
                failed.append(res)
            if done % 500 == 0:
                print(done, "of", len(jobs), flush=True)
    print(f"finished: {len(jobs) - len(failed)} ok, {len(failed)} failed")
    for f in failed[:20]:
        print(" ", f)


if __name__ == "__main__":
    main()
