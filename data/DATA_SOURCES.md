# Data Sources and Provenance

This page lists where every dataset comes from, what format it's in, and what I found when I checked it. I downloaded everything on **2026-10-20** and checked it in `notebooks/01_data_audit.ipynb`.

My analysis window is **2016-07-26 to 2026-06-29** (87,024 hours). DSCOVR's archive ends on 2026-06-29, while the other sources go on until October 2026. I never edit the raw files. The cleaning and alignment steps are in the notebooks (`scripts/download_data.py` re-downloads the raw files).

| What | Provider | Where it comes from | Where it is on disk | Format | Size and records |
|---|---|---|---|---|---|
| Solar wind (proton speed, density, temperature), Faraday cup, 1 minute (`f1m`) | NOAA NCEI, DSCOVR | https://archive.data.noaa.gov/satellite-spaceweather/ (prefix `DSCOVR/DSCOVR/FC/f1m/`); portal: https://www.ncei.noaa.gov/cloud-access/space-weather-portal/ | `raw/dscovr/f1m/YYYY/MM/*.nc.gz` | NetCDF, gzip, one file per day | 169 MB; 3,644 files; 3,575 different days |
| Magnetic field (Bx, By, Bz, Bt), magnetometer, 1 minute (`m1m`) | NOAA NCEI, DSCOVR | same archive (prefix `DSCOVR/DSCOVR/MAG/m1m/`) | `raw/dscovr/m1m/YYYY/MM/*.nc.gz` | NetCDF, gzip, one file per day | 174 MB; 3,645 files; 3,575 different days |
| Kp, ap, Ap, sunspot number, F10.7 | GFZ Potsdam | https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt | `raw/kp/gfz_Kp_ap_Ap_SN_F107_since_1932.txt` | text, one row per day with 8 three-hour slots | 5.5 MB; 34,626 days (1932-01-01 to 2026-10-19) |
| Kp, ap, F10.7 | CelesTrak | https://celestrak.org/SpaceData/SW-All.csv | `raw/kp/celestrak_SW-All.csv` and `raw/f107/celestrak_SW-All.csv` (the same file twice) | CSV, daily | 2.9 MB; 25,444 rows (1957-10-01 to 2041-10-01, including predictions) |
| Dst, hourly (best available version) | WDC for Geomagnetism, Kyoto (HAPI `hour_dst`) | https://wdc.kugi.kyoto-u.ac.jp/hapi/ | `raw/dst/kyoto_hour_dst_YYYY.csv` and `raw/dst/kyoto_hour_dst_info.json` | CSV with no header: time, Dst (nT), version code | about 1 MB; 90,327 hours (2016-07-01 to 2026-10-20) |
| Dst, Kp, ap, F10.7, solar wind and field (hourly, for cross-checks) | NASA OMNIWeb (OMNI2) | https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/ | `raw/dst/omni2_hourly_YYYY.dat` and `raw/dst/omni2_format_description.txt` | fixed-width text, 55 columns, no header | about 35 MB; 96,432 hours (2016-01-01 to 2026-12-31, padded to full years) |

I also keep two cache files that the notebooks rebuild from the raw data (they're not committed): `interim/dscovr_mag_1min_all.parquet` and `interim/dscovr_fc_1min_all.parquet`.

## Things to know about the formats
- **DSCOVR:** time is stored as milliseconds since 1970-01-01 UTC. I use the GSM components. The magnetometer's 1-minute values use a Hodges-Lehmann estimator and the Faraday cup uses a Huber estimator. Missing minutes are NaN, and they match `fill_flag` = 1 exactly.
- **GFZ:** missing values are coded as -1 (only F10.7 has any). The quality flag `D` means 0 = Kp and SN preliminary, 1 = Kp definitive, 2 = both definitive.
- **CelesTrak:** Kp is stored as Kp times 10, rounded (87 means 8.667).
- **Kyoto:** timestamps are the middle of each hour in UTC (HH:29:30). Version code 20 = final, 10 = provisional, 0 = real-time. The fill value 99999 doesn't appear anywhere in what I downloaded.
- **OMNI:** the hour label is the start of the hour (I checked: Dst of -406 at 02:00 matches Kyoto's 02:29:30 value). Each field has its own fill values (999.9, 9999., 99999 and so on). OMNI's `f107` is the adjusted (1 AU) value.

## What I found in each source (details in notebook 01)

### DSCOVR magnetometer
- After keeping the latest version of the 64 days that have more than one file, there are 3,575 different days, and 51 days are missing altogether. That's 5,148,000 minute rows.
- 570,455 minutes (11.08%) are NaN inside the files. Counting the missing days too it's 643,895 of 5,221,440 (12.33%).
- 19 gaps longer than 24 hours hold 66% of the missing minutes. The main outages are 2019-06-27 to 2020-02-06 (July to December 2019 are 88 to 100% missing) and 2025-07-15 to 2025-10-02 (79 days). Early 2026 is also poor (January 42%, February 22%, March 14%). The files don't say why.
- `overall_quality` tells me nothing (it's 0 even on empty rows). `maneuver_flag` covers 15,583 minutes and `possible_saturation_flag` just 1 (2017-05-12 15:18, a quiet minute).
- `bt` isn't always the magnitude of the three components (the median difference is 0.012 nT, and 154 minutes differ by more than 5 nT, mostly when a shock arrives or a storm starts). I prefer the components.
- The values are physically sensible: Bt from 0.12 to 95.2 nT and Bz from -57.9 to +83.3 nT. All the extremes fall in one stretch, 2026-01-19 19:02 to 21:31 UTC.
- Once the timing is accounted for it agrees well with OMNI's hourly field: the Bz correlation is 0.718 with no delay and 0.909 with a 60-minute delay, and for the total field 0.956 and 0.987.

### DSCOVR Faraday cup
- 3,575 different days (the same as the magnetometer) and 5,148,000 rows. There are 83 data variables and 63 of them are flags. `large_flow_angle_flag` doesn't exist in the 260 files before 2017-04-13.
- 624,747 minutes (12.14%) are NaN. The outages are the same as the magnetometer's but a bit worse (2019: 50.9%, 2025: 24.6%).
- **The alpha-particle data is completely empty.**
- `reduced_proton_quality_flag` is set on 27% of minutes and goes up with the solar cycle (5 to 16% in 2016 to 2021, 35% in 2022, 56% in 2023, 53% in 2024, 43% in 2025 to 2026).
- The valid ranges declared in the files aren't enforced: speed goes up to 1,159.7 km/s (435 minutes above 1,111), density from 0.05 to 161.6 cm^-3, and temperature from 1,103 to 7.3 million K, with 38,404 minutes exactly at 2,000 K.
- **It fails in fast wind.** Compared with OMNI's hourly speed, the error grows with the share of flagged minutes (hours off by more than 100 km/s: 0% when unflagged, 7.2% when 75 to under 100% are flagged, 25.5% when all are flagged). In the 471 hours where every minute is flagged and OMNI speed is over 700 km/s, DSCOVR's speed averages 485 km/s (middle half 465 to 498), a flat plateau. In May 2024, from about 14:00 UTC on 11 May every hour is fully flagged and DSCOVR shows about 470 to 510 km/s while OMNI shows 770 to 1,006 km/s, for at least 34 hours. The values are there (no NaN), so a check for gaps doesn't catch them. I haven't confirmed the cause.

### GFZ
- No duplicate or missing dates. Kp, ap, Ap and SN are complete in every year. F10.7 starts on 1947-02-14 (6,178 missing days, mostly before that), and there are 9 isolated gap days inside my window (2020-10-23, 2020-12-17 and 18, 2021-05-09, 2021-06-16, 2024-08-26, 2025-02-12, 2025-02-17 and 2026-05-09).
- Kp is definitive up to 2026-09-30 and only preliminary from 2026-10-01 to 2026-10-19. May 2024 is definitive.
- Storm days are rare: 318 of 3,738 days (8.5%) reach G1 or above, and one day reaches Kp 9 (2024-05-11).

### CelesTrak
- The observed record ends on 2026-10-19. 45 daily `PRD` rows (2026-10-20 to 2026-12-03) hold **forecast Kp values** with a repeating pattern, and 178 monthly `PRM` rows have no Kp. I drop all `PRD` and `PRM` rows.
- Kp matches GFZ in all 29,904 slots of my window (biggest difference 0.033, which is just rounding), and Ap is identical.
- F10.7 matches GFZ on 3,714 of 3,729 shared days. 15 days differ by 0.2 to 2.9 sfu (8 of them from 2018-02-20 to 2018-02-28), and both sources mark them as final. I don't know why.
- `INT` rows are interpolated F10.7. CelesTrak fills all 9 of GFZ's gap days this way.

### Kyoto Dst
- 90,327 hourly rows with no duplicates and no missing hours. The range is -406 nT (2024-05-11 02:29 UTC) to +71 nT. Hours at or below -50: 2,570; at or below -100: 363; at or below -250: 30.
- Final from 2016-07-01 to 2020-12-31, provisional from 2021-01-01 to 2026-09-30, real-time from 2026-10-01. **May 2024 is provisional.**
- The daily minimum Dst and the daily maximum Kp correlate at -0.72. Days where Dst is still low after Kp has dropped are real recovery days, not errors.

### OMNI
- For 2016-07-01 to 2026-10-20 there are 90,336 hours. The most recent data lags: the field ends on 2026-09-30, plasma on 2026-10-13, Kp, ap and f107 on 2026-10-15, and Dst on 2026-10-16 14:00.
- Dst matches Kyoto in 90,229 of 90,231 hours (the 2 provisional hours that differ are off by 1 to 2 nT). Kp and ap match GFZ in every hour I compared. OMNI copies its Dst from Kyoto, so these checks only show my parsing is right, not that the values are accurate.
- F10.7 is the adjusted value. 76 days differ from CelesTrak's `F10.7_ADJ` by more than 0.1 sfu, and 53 days differ from GFZ's `F10.7adj`.
- It has almost no field gaps in 2019 (0.0% missing B), while the DSCOVR magnetometer was about 50% missing that year.

## How much usable data there is
Hours with Dst, magnetometer and plasma all available ("core hours"): 85.6% from DSCOVR alone, and 86.3 to 86.4% with the OMNI plasma fallback. The magnetometer limits it (86.6% of hours). By year: 2019 is 46.6%, 2025 is 64.4% (68.5% with the fallback), 2020 is 80.5%, and the other years are 92 to 98%.

## Licences and credit (still to check)
I haven't checked these against each provider's current terms yet. This is what I know so far:
- NOAA NCEI / DSCOVR: US Government data. Cite NOAA NCEI.
- GFZ Kp: CC BY 4.0. Cite Matzka et al. (2021), GFZ German Research Centre for Geosciences.
- CelesTrak: free to use. Cite CelesTrak (derived from GFZ and NOAA).
- Kyoto WDC Dst: cite Nose et al. (2015), doi:10.17593/14515-74000. Their terms (https://wdc.kugi.kyoto-u.ac.jp/wdc/Sec3.html, read 2026-10-26) say the data and services are available for scientific use without restrictions, ask you to acknowledge the data suppliers and WDC Kyoto and include the data DOIs, and don't allow commercial applications. They don't say whether the web pages or documents can be redistributed.
- NASA OMNI: public. Cite King and Papitashvili (2005), OMNIWeb.

I need to check each of these before I submit the report.
