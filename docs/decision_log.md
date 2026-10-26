# Decision Log

This is where I keep track of the choices I made on this project and why. Each one has a status:
- **Agreed**: decided and settled.
- **Adopted**: I'm going with it and I'm happy to change it.
- **Proposed**: my recommendation, not confirmed yet.
- **Open**: still needs a decision.

All dates are 2026-10-20 unless the table says otherwise (the audit and everything after it was done on 2026-10-25).

## 1. Scope and approach

| # | Decision | Status | Why |
|---|---|---|---|
| 1 | Build a system that explains events (find them, describe them, retrieve background text), not a forecasting model | Agreed | My main success measure is how well the generated explanation is backed up by evidence. The proposal has no forecasting step, and forecasting alone would use up all my time. |
| 2 | Look at historical events only, with no live data feed. The pipeline works on a window of aligned data, so a live feed could be added later | Agreed | The proposal says the data is already downloaded. I need events with known answers to evaluate against. A live feed would be extra work that doesn't help the main metric. |
| 3 | Explore and report in notebooks, then move the stable logic into `src/spacewx_rag/` with small scripts in `scripts/` | Agreed | The app and anyone checking my work need code they can run again. The notebooks feed the report. |
| 4 | Use a Python 3.11 virtual environment inside the project folder, and I do all the git work myself | Agreed | The system Python (3.9) is too old for the packages. |
| 5 | Keep this project fully separate from Space Weather Dashboard V2 (no shared code or data) | Agreed | They're different projects. |

## 2. Data audit (notebook 01)

| # | Decision | Status | Why |
|---|---|---|---|
| 6 | Take Kp and ap from GFZ and don't use CelesTrak's Kp | Proposed | CelesTrak's Kp and Ap are the GFZ values rescaled (biggest difference 0.033). |
| 7 | Take F10.7 from CelesTrak's adjusted column (observed rows plus interpolated days, which are flagged). Keep GFZ and OMNI as cross-checks | Proposed (still open) | The proposal names CelesTrak for F10.7. OMNI's `f107` is the adjusted value. CelesTrak and GFZ differ by up to 2.9 sfu on 15 days (0.4% of days) and I don't know why. Nine gap days are interpolated. Both the adjusted and observed columns are stored, so switching later needs no rebuild. |
| 8 | Drop CelesTrak's `PRD` and `PRM` prediction rows | Proposed | The `PRD` rows hold repeating forecast Kp values, not real observations. |
| 9 | Use Kyoto Dst as the main Dst, OMNI Dst only as a cross-check, and keep Kyoto's version code as a quality flag | Proposed | OMNI copies its Dst from Kyoto, so it isn't independent (90,229 of 90,231 hours are identical). |
| 10 | Accept provisional Dst (2021 to September 2026, including May 2024) and state it as a limitation | Open | Final Dst covers only 44.7% of the hours. The headline storm is provisional, so Kyoto may still revise it. |
| 11 | Magnetometer rule: NaN or `fill_flag` means missing; blank the maneuver and saturation minutes (15,584 minutes); keep the other flags; use the components and not `bt`; ignore `overall_quality` | Proposed | `overall_quality` is 0 even on empty rows. Flagged minutes don't have more extreme values than unflagged ones. `bt` differs from the vector magnitude when the field changes fast. |
| 12 | DSCOVR plasma: hours where 75% or more of the minutes are flagged as reduced quality use OMNI hourly speed, density and temperature instead, with a source column | Agreed (25 Oct) | I compared the options (3,107 hours differ from OMNI by more than 100 km/s). A 75% threshold replaces 23.2% of hours, removes 95.9% of the bad hours and leaves 127. A 100% threshold would leave 661 bad hours. Going below 75% removes at most 3.0 more points of bad hours (98.9% at 25%) but replaces 6.6 to 14.5 points more hours. Replaced hours get OMNI values, so the cost is resolution and independence, not accuracy. In May 2024 DSCOVR shows about 480 km/s for 34+ hours while OMNI reaches 1,006. |
| 13 | Drop the alpha-particle variables | Proposed | `alpha_density` is empty for the whole record. |
| 14 | Fill magnetometer outages from OMNI hourly B and Bz, with a source column. DSCOVR is delayed first, and the OMNI values are used as they are (already delayed to Earth) | Agreed (25 Oct) | In 2019 only 46.6% of the core hours are usable, in 2025 64.4%. OMNI's B and Bz are valid in 99.5% of hours. The cost is losing 1-minute resolution in filled hours. |
| 15 | Check that using OMNI plasma and field data fits the proposal's rule that the dataset is locked | Open | OMNI is already a listed source (for Dst). I went ahead with the OMNI fallback (decisions 12 and 14) but I still want to check with the programme. |
| 16 | Line DSCOVR up with the Earth-based indices using a speed-dependent delay (distance divided by speed), not a fixed one | Adopted (25 Oct) | The best fixed delay is about 60 minutes (Bz correlation 0.718 to 0.909) but it's about 35 to 40 minutes at 700 km/s. In storms DSCOVR's own speed is unreliable, so OMNI speed is used for the delay. |
| 17 | Define events from Dst, with a window that includes the recovery, and not from calendar days or Kp alone | Adopted (25 Oct) | 7 of the 49 days with Dst at or below -100 have a Kp maximum under 6, mostly recovery days. The settings are in decisions 23 to 26. |
| 18 | Evaluate on 15 events: 1 super (E216, 2024-05-11), 5 intense (E158, E231, E058, E027, E256), 4 moderate (E002, E095, E117, E039) and 5 quiet (E078, E105, E143, E121, E221), with lighter references. Saved in `evaluation/reference_events/evaluation_set.csv` | Agreed (25 Oct) | Storms are rare (only 8.5% of days reach G1), so I picked events on purpose. 35 of the 49 days with Dst at or below -100 have at least 90% DSCOVR Bz coverage, 12 partial and 2 below 50%. Caveats are in decision 27. |

## 3. Putting the data together (notebook 02)

| # | Decision | Status | Why |
|---|---|---|---|
| 19 | Delay = 1.5 million km divided by speed, applied to the DSCOVR minute data before averaging to hours. Speed is DSCOVR's where it's reliable and OMNI's otherwise. OMNI values are used as they are | Adopted | Against OMNI's hourly Bz the correlation is 0.919 (0.909 with a fixed 60 minutes, 0.718 with no delay). Of the distances I tried (1.0 to 1.8 million km), 1.5 fits best, which is the real distance to L1. It's better in every speed range. In fast wind (over 600 km/s, 4,888 hours) the median error is 0.600 nT instead of 0.828. |
| 20 | Store F10.7 adjusted and observed, plus a flag for interpolated days and a flag for spikes (adjusted value more than 1.3 times the 7-day median). I don't change the values | Adopted | There are one-day spikes (2024-07-30 at 412.9 between 230.0 and 242.2, and 2023-02-17 at 335.0 between 159.2 and 163.3). Nine days are flagged. I haven't checked the cause, and the 1.3 factor is just a rule of thumb. |
| 21 | One hourly table at `data/processed/aligned_hourly.parquet`: 87,024 rows, 20 columns, UTC, with source columns for field and plasma | Adopted | Field: 86.6% from DSCOVR, 13.4% OMNI, 0.0% missing. Plasma: 64.7%, 35.0%, 0.3%. `bx`, `by` and `bz_min` only exist where DSCOVR was the source. |

What the exploratory analysis showed, which matters later:
- Driver features for an event should look back at least about 12 hours. Dst correlates -0.758 with the 12-hour mean of vBs, and it stays flat at 24 hours.
- Intense storms are bunched up: 88% of the 355 hours with Dst at or below -100 are in 2023 to 2026, 38% in 2024, and none in 2019 to 2020. Seasonal patterns are weak over ten years.
- For strong drivers the Dst response is very scattered (about -245 to -390 nT at a 12-hour mean vBs near 15 mV/m).

## 4. Defining events (notebook 03)

| # | Decision | Status | Why |
|---|---|---|---|
| 22 | The event set includes both storms (found from Dst, with a window covering the 12-hour look-back and the recovery) and quiet windows to use as controls | Agreed (25 Oct) | The quiet windows let me check that the system calls calm conditions calm and doesn't make drama up. The details are in decisions 23 to 26. |
| 23 | Storm: a run of hourly Dst at or below -50 nT. Runs less than 12 hours apart are merged. The window starts 12 hours before the run and ends when Dst is back above -30 nT (at most 72 hours after the run). Classes by lowest Dst: moderate (-50 to -100), intense (-100 to -250), super (below -250) | Adopted (25 Oct) | With a -50 threshold there are 351 raw runs, then 228, 202 and 175 after merging at 6, 12 and 24 hours. At -100 it's 48, 36, 33 and 31. At -250 it's 3, 2, 2 and 2. A 12-hour gap cuts the flicker without stretching events (longest window 148 hours). Result: 202 storms (171 moderate, 29 intense, 2 super). Windows have a median of 30 hours and a mean of 38. |
| 24 | Quiet window: 48 hours with Dst above -30 and Kp never above 3, at least 24 hours from any storm window, field at least 90% from DSCOVR, plasma present in at least 95% of the hours. I sampled 100, balanced across F10.7 levels (34 below 100, 33 at 100 to 150, 33 above 150), random seed 42 | Adopted | There were 520 candidates out of 1,813 blocks (386, 67 and 67 by level), and every year is covered (2019 has 2 windows, 2025 has 3). I allow plasma from OMNI because DSCOVR's reduced-quality flag covers over 50% of minutes in 2023 to 2025. |
| 25 | Keep marginal storms (core of 3 hours or less and lowest Dst above -60 nT; 50 of the 202) in the table with a `marginal` flag, and leave them out of the evaluation by default | Adopted | They meet the Dst rule but they're brief. The flag means the app can still describe them. |
| 26 | Keep overlapping windows (29 pairs of storms) as separate events with `overlaps_prev` and `overlaps_next` flags. The evaluation prefers events that don't overlap, but May 2024 (which overlaps the next event) stays as the headline case | Adopted | The overlaps are real storms one after another. Both super storms are affected: May 2024 overlaps the next event, and October 2024 overlaps the previous one and has 82% field coverage. |

## 5. Evaluation set and notebook plan

| # | Decision | Status | Why |
|---|---|---|---|
| 27 | The 15 evaluation events (decision 18) go ahead with their caveats written down | Agreed | E216 (the headline storm) has only 29% of its plasma from DSCOVR (the rest is OMNI hourly) and its window overlaps the next event, so I check its plasma claims against OMNI. Only 5 of the 15 events have a final Dst value at the minimum. The other 10 (2021 onward) use provisional Dst. I picked quiet windows with at least 50% DSCOVR plasma, because asking for 90% left only 20, 5 and 1 windows per F10.7 level. E105, E143 and E121 have Kp right at the limit of 3. E143 is deliberately the quiet window with the most southward Bz (-7.1 nT). |
| 28 | Notebook 04 is now `04_event_cards`. It builds structured event cards (facts the model may state, where each value came from, caveats, and claims that need a retrieved source) and doesn't extract features again | Agreed | The features were already computed in notebook 03 (21 per event). The cards link the data to the retrieval step and make groundedness measurable. |

## 6. Event cards (notebook 04)

| # | Decision | Status | Why |
|---|---|---|---|
| 29 | A card holds facts plus hints that are computed by rules (like "southward Bz for 9 hours" or "fast solar wind"). Physical causes and impacts can only come from retrieved documents, never from the card | Agreed (25 Oct) | The hints give retrieval better questions and can be checked against the data. The thresholds are rules of thumb, listed in `docs/event_definitions.md`. |
| 30 | Every card carries caveats about hourly means, the Earth-arrival time shift and the sign convention for Bz, plus caveats about data quality when they apply | Adopted (25 Oct) | The average number of caveats per card is 4.6 for quiet, 5.3 moderate, 5.2 intense and 6.5 super. |
| 31 | A card's Dst minimum is the lowest value inside the event's own core. When the window holds a deeper value from an overlapping event, the card says so | Adopted | E159 (-51 against -52 nT from E160) and E282 (-65 against -74 nT from E283). I found them when checking the cards against the aligned table. Neither is in the evaluation set. |
| 32 | Remove `evaluation_role` from the cards before they go to the model in notebooks 06 and 07 | Agreed | The label would show which events are in the evaluation set. |

## 7. Knowledge base and retrieval (notebook 05)

| # | Decision | Status | Why |
|---|---|---|---|
| 33 | The knowledge base holds general reference documents (storm scales, Kp and Dst definitions, solar wind and Bz physics, impacts) plus a few documents about May 2024 only | Agreed (25 Oct) | This tests two things: grounded cause and impact statements for the headline storm, and the system saying "no supporting source" for the other 14 events. Only public-domain or open-licence sources, with the licence recorded for each. |
| 34 | 21 documents downloaded to `knowledge_base/raw_docs/` (7.3 MB) with a manifest of source, licence status and role. Document 10 (the Kyoto home page) is left out of the index | Agreed (25 Oct) | They are 14 NOAA documents (13 pages and the scales PDF, public domain under NWS policy unless noted), GFZ's "About Kp", Kyoto's "On the Dst index" and version definition, a NASA tutorial, the Hayakawa et al. 2025 paper (CC BY 4.0) and Wikipedia (CC BY-SA 4.0). I couldn't find an official NOAA report on May 2024, and NOAA has no pages on pipeline or aviation impacts. Kyoto's terms allow scientific use with credit (no commercial use) but don't say whether its pages can be redistributed, so those three are included for academic use with attribution. I haven't verified the NASA document's licence. |
| 35 | Embed the text on my own machine with sentence-transformers, using BAAI/bge-small-en-v1.5 (about 130 MB, downloaded the first time), and store it in Chroma | Agreed (25 Oct) | No API key, no cost and the results are repeatable. It fits the open-source stack in the proposal. The defaults are in `.env.example`. |
| 36 | Decide which LLM writes the descriptions (notebook 06) later | Open | The cards and the retrieval don't depend on it. Options: the OpenAI API, the Anthropic API, or a local model through Ollama. |
| 37 | Add `beautifulsoup4` and `pypdf` to `requirements.txt` to read the documents, and regenerate the lock file | Adopted | Means I don't depend on a system PDF tool. |
| 38 | Chunking: split on the `##` headings, then a recursive splitter at 900 characters with 120 overlap, drop pieces under 120 characters, and add the title in front when embedding | Adopted (25 Oct) | This gives 290 chunks from 20 documents (average 715 characters). The Hayakawa paper makes 122 of them and the two May 2024 documents 137 in total. A small flaw: some chunks start with a stray ". " left over by the splitter. |
| 39 | Index: BAAI/bge-small-en-v1.5 (384 dimensions, normalised), a Chroma collection called `spaceweather_kb`, cosine similarity, the bge query prefix. Stored in `knowledge_base/vector_store/` (not committed) | Adopted | 290 chunks indexed. |
| 40 | The May 2024 documents (12 and 13) can only be retrieved for events whose window overlaps 2024-05-08 to 2024-05-20 | Adopted | None of the other 14 evaluation events got them, and E216 does. |
| 41 | Search with several targeted queries per event (physics, impacts at that storm level, what the Kp and Dst values mean, plus provisional versus final Dst when the card says provisional). In notebook 06, impact claims get one query per impact area | Adopted | One query based on the drivers returned only the storms page and the scales for all 14 non-May events, and never the impact pages. |
| 42 | Don't use the retrieval similarity score to decide "no supporting source". Do a claim-by-claim check against the retrieved text in notebook 06 | Adopted | Questions with no real source scored 0.47 to 0.79 (aviation radiation 0.785, stock market 0.759, Carrington cost 0.606, capital of France 0.472), which overlaps the top-1 range of the real questions (0.749 to 0.854). |
| 43 | Retrieval quality on 18 hand-labelled questions: hit@1 0.78, hit@3 1.0, hit@5 1.0, MRR 0.889 | Recorded, labels still to be reviewed | With the labels as first written the numbers were 0.72, 0.94, 0.94 and 0.833. One label (spacecraft charging) was corrected after I read the documents (documents 1 and 2 mention surface charging, document 8 doesn't), so the corrected figures aren't independent evidence. The set is small and needs a proper review before I quote it. Results are in `evaluation/results/retrieval_questions.csv` and `retrieval_events.csv`. |

## 8. Mistakes I found and fixed along the way
- I first assumed CelesTrak's `PRD` rows had no Kp. They do (they're forecasts). Fixed in `data/DATA_SOURCES.md`.
- I compared OMNI's `f107` with CelesTrak's observed F10.7, but it's the adjusted value.
- The Faraday cup files before 2017-04-13 don't have the column `large_flow_angle_flag`, so the loader now copes with missing columns.
- One figure in notebook 01 was saved blank, because the save command was in a separate cell after the plot. I moved it into the plot cell and regenerated the figure.
