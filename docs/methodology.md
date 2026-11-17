# Methodology

This describes how I'm approaching the project and what I've done so far. Everything up to the evaluation is built; the Streamlit app is next.

## 1. Objective
I want to turn data about past space weather events into clear, structured explanations that are backed up by evidence. The system describes an event from the observations, pulls supporting scientific background from a knowledge base, and writes an explanation that sticks to those two sources. It doesn't forecast anything.

My main success measure is groundedness: the share of claims that the input observations or the retrieved sources support. I'll also look at retrieval relevance, completeness, the rate of unsupported claims, and consistency.

## 2. Data
I use six sources: the DSCOVR magnetometer and Faraday cup (NOAA NCEI), Kp and ap (GFZ), F10.7 (CelesTrak), Dst (Kyoto WDC), and NASA OMNI, which I use to cross-check and to fill gaps. Where each one comes from, its format and what I found when checking it are in `data/DATA_SOURCES.md`.

The analysis window runs from 2016-07-26 to 2026-06-29 (87,024 hours). It starts when DSCOVR began operating and ends where its published archive ends.

| What | Source | How often |
|---|---|---|
| Magnetic field Bx, By, Bz (GSM) and Bt | DSCOVR magnetometer | every minute |
| Proton speed, density, temperature | DSCOVR Faraday cup (OMNI as backup) | every minute (OMNI is hourly) |
| Dst | Kyoto WDC | every hour |
| Kp, ap | GFZ | every 3 hours |
| F10.7 (adjusted) | CelesTrak | every day |

## 3. Data cleaning rules
These came out of the audit (the reasons are in `decision_log.md`).
- Missing values become NaN. That means -1 for F10.7 in the GFZ file, the fill values in each OMNI field, and for DSCOVR the NaN values (which match `fill_flag` exactly).
- If a DSCOVR day has several versions of its file, I keep the latest one.
- Magnetometer: I blank the maneuver and saturation minutes, use the GSM components, and don't rely on `bt` or `overall_quality`.
- Faraday cup: I drop the alpha variables because they're empty. Hours where at least 75% of the minutes are flagged as reduced quality count as unreliable, and I use OMNI hourly values for them instead, with a source column.
- I drop CelesTrak's prediction rows and flag the interpolated F10.7 days.
- I keep Kyoto's version code (final, provisional or real-time) as a quality flag.

## 4. Lining the data up in time (done, notebook 02)
The result is `data/processed/aligned_hourly.parquet`: 87,024 hourly rows (2016-07-26 00:00 to 2026-06-29 23:00 UTC) and 20 columns.
- Everything is on an hourly UTC timeline. Kyoto's timestamps sit in the middle of each hour (HH:29:30) so I round them down to the hour. OMNI's hour labels are the start of the hour. Kp and ap are held over their 3-hour slot and the daily F10.7 over its day.
- DSCOVR is upstream of Earth, so I shift its minute data to the time it reaches Earth: delay = 1.5 million km divided by the solar wind speed. Then I average into hours (an hour needs at least 45 valid minutes). The speed is DSCOVR's where that's reliable, and OMNI's otherwise.
- Against OMNI's hourly Bz the correlation is 0.919. It's 0.909 with a fixed 60-minute delay and 0.718 with no delay. The improvement shows up in slow, medium and fast wind, and it's biggest in fast wind (over 600 km/s).
- Plasma: DSCOVR hours where at least 75% of the minutes are flagged use OMNI hourly speed, density and temperature. Field: gaps in the DSCOVR data are filled with OMNI hourly Bz and Bt. OMNI values are already delayed to Earth, so I don't shift them again. Every hour records where its value came from (`bz_src`, `plasma_src`).
- That gives these sources: field 86.6% DSCOVR and 13.4% OMNI; plasma 64.7% DSCOVR, 35.0% OMNI and 0.3% missing. `bx`, `by` and `bz_min` only exist where DSCOVR was the source (86.6%).
- F10.7: I store the adjusted and the observed values, with a flag for interpolated days (9) and a flag for spikes (9 days above 1.3 times the 7-day median). I don't change the values.

## 5. Finding events and their features (done, notebook 03)
The full definitions are in `docs/event_definitions.md`, and the output is `data/events/event_table.parquet`.
- Storms come from Dst: runs at or below -50 nT, merged when less than 12 hours apart, with a window from 12 hours before the run until recovery (Dst back above -30 nT, at most 72 hours later). They're classed by lowest Dst as moderate, intense or super. That gives 202 storms (171 moderate, 29 intense, 2 super).
- The quiet control windows (100 picked from 520 candidates) are 48-hour blocks with Dst above -30 nT and Kp at most 3, kept away from storms and spread across solar activity levels.
- For each event I work out timing, Dst, Bz, the solar wind electric field, speed, density, pressure, Kp, ap, F10.7 and data quality. Quiet and moderate events separate clearly (for example Bz minimum -3.5 against -8.4 nT, Kp 2.3 against 5.3). Speed doesn't separate moderate from intense storms, but Bz does.
- 50 of the storms are marginal (short and only just past the threshold) and 29 storm windows overlap a neighbour. Both are flagged.

## 6. Knowledge base and retrieval (done, notebook 05)
- The corpus is 21 downloaded documents (7.3 MB) in `knowledge_base/raw_docs/`, with a manifest of source, licence status and role. 20 are indexed (one is a service home page, so I left it out). They are 14 NOAA SWPC documents, GFZ (Kp), Kyoto WDC (Dst description and version definitions), a NASA tutorial, the Hayakawa et al. 2025 paper on May 2024 (CC BY 4.0) and the Wikipedia summary (CC BY-SA 4.0). The scope is general references plus May 2024 only.
- Processing: I extract the text for each type of source, split it into 290 chunks (about 900 characters, 120 overlap, keeping section headings as metadata), embed them on my own machine with BAAI/bge-small-en-v1.5 and index them in Chroma with cosine similarity.
- Retrieval: for each event I run several searches built from its card (physics, impacts at that storm level, what the index values mean, and provisional versus final Dst when it applies). The May 2024 documents are only returned for events that overlap 8 to 20 May 2024.
- Quality: on 18 hand-labelled questions I get hit@1 0.78, hit@3 1.0, hit@5 1.0 and MRR 0.889 (0.72, 0.94, 0.94 and 0.833 with the labels as first written; one label was corrected after I read the text). The set is small and I still need to review it properly before quoting these numbers as a result.
- Limits: the similarity score can't tell supported questions from unsupported ones on related topics (unsupported questions scored 0.47 to 0.79 and real ones 0.749 to 0.854). So "no supporting source" has to come from a claim-by-claim check when the text is generated. Outside May 2024 the background I retrieve is about the storm class, not the specific event. I couldn't find an official NOAA report on the May 2024 event.

## 7. Generation (done, notebook 06)
- For each event the prompt holds the card (without the evaluation label) and 15 to 18 de-duplicated passages from targeted queries: physics, one per impact area for storms (power, satellites, GNSS, HF radio, aurora), what the index values mean, and provisional versus final Dst when it applies. The rules are in `prompts/system.md`: facts only from the card, causes and impacts only with a cited passage, impacts as "could" at the card's storm level, quiet windows described as calm, caveats repeated, and "no supporting source" for anything else.
- The output is JSON in a fixed format (summary, characteristics with card evidence, possible physical interpretation and potential impacts with chunk ids, caveats, unsupported or unknown). For each event the schema limits the evidence to real card fields and the citations to the retrieved chunks.
- Two generators: an offline extractive baseline (no model) and `qwen2.5:7b-instruct` running locally through Ollama at temperature 0. Also a no-retrieval ablation (card only) and a consistency run (five events, three runs each at temperature 0.7).

## 8. Evaluation (done, notebook 07)
- Every claim is checked: numbers and written dates against the card (or the cited passages for causes and impacts), no cause or impact language in the card-only sections, real evidence paths and citations, no forecasts, no impacts for quiet windows, and an LLM judge for whether the cited passages support each cause or impact. The judge must quote the supporting sentence (checked against the passages), and code checks that an impact isn't taken from a higher NOAA G level than the event's.
- Metrics on the 15 reference events: groundedness (main), unsupported-claim rate, completeness against hand-written reference points, must-not violations, consistency, and retrieval relevance on 42 labelled questions (24 new ones labelled before retrieval was run). A manual audit of the 21 cause and impact claims of the main run measures how far the judge can be trusted.
- Main results: groundedness 0.935 with retrieval (0.920 after the manual audit) against 0.825 without; completeness 0.86 against 0.75; no number errors; no impacts or storm language for quiet windows. The full figures are in notebook 07 and the README.

## 9. Limitations so far
- The DSCOVR magnetometer and plasma instruments have long outages (2019-06 to 2020-02, 2025-07 to 2025-10, and early 2026), and the archive ends on 2026-06-29.
- The DSCOVR Faraday cup gives wrong speeds in fast solar wind. In May 2024 that lasted at least 34 hours from about 14:00 UTC on 11 May. I haven't confirmed the cause. The values are there but they're wrong, so a check for gaps doesn't catch them.
- Dst from 2021 onward is provisional, including May 2024, so it may be revised.
- F10.7 differs between sources on 15 days (by up to 2.9 sfu) and nine days are interpolated. I don't know why the sources disagree.
- OMNI's Dst, Kp and ap are copies of the primary sources, so agreement only shows my parsing is right, not that the values are accurate. OMNI's plasma and field data are hourly and already moved to Earth's bow shock.
- There's no alpha-particle data.
- The quality flags in the DSCOVR files are only partly documented (for example `overall_quality` and `normal_mode_flag`), so how I use them is based on what I saw in the data.
- Storms are rare (only 8.5% of days reach Kp 5 or more), so I have to choose the evaluation events carefully.
