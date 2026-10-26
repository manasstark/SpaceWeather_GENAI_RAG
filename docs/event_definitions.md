# Event Definitions

This page explains exactly how I define an event. It's built in `notebooks/03_event_detection.ipynb` from `data/processed/aligned_hourly.parquet`. The output is `data/events/event_table.parquet` and `.csv` (302 events, 33 columns). The reasons behind each choice are in `docs/decision_log.md` (decisions 22 to 26).

## Storm events (202)
- **Core:** a run of hourly Dst values at or below -50 nT. If two runs are less than 12 hours apart, I merge them into one event.
- **Window:** it starts 12 hours before the first hour of the core, because Dst correlates -0.758 with the 12-hour mean of vBs, so the driver before the storm matters. It ends at the first hour after the core where Dst is back above -30 nT (and never more than 72 hours after the core ends).
- **Class, by the lowest Dst:** moderate (-50 to -100 nT): 171; intense (-100 to -250 nT): 29; super (below -250 nT): 2.
- **Size:** windows have a median of 30 hours, a mean of 38 and a maximum of 148. The core has a median of 7 hours.
- **`marginal` flag (50 events):** the core lasts 3 hours or less and the lowest Dst is above -60 nT. These stay in the table but I leave them out of the evaluation by default.
- **Overlaps (29 pairs):** sometimes one storm's window overlaps the next one's. I keep them as separate events and mark them with `overlaps_prev` and `overlaps_next`.

How the number of events changes with the settings: at -30 nT there are 1,108 raw runs, or 503 after a 12-hour merge. At -50 nT there are 351 raw runs, then 228, 202 and 175 after merging at 6, 12 and 24 hours. At -100 nT it's 48, 36, 33 and 31, and at -250 nT it's 3, 2, 2 and 2.

## Quiet windows (100 picked from 520 candidates)
- 48-hour blocks where Dst stays above -30 nT and Kp never goes above 3.
- At least 24 hours away from any storm window.
- At least 90% of the magnetic field from DSCOVR, and plasma present (from any source) in at least 95% of the hours.
- Balanced across solar activity: 34 windows with mean F10.7 below 100, 33 between 100 and 150, and 33 above 150 (random seed 42). Every year has some, but 2019 and 2025 only have 2 and 3.

## What I work out for each event (from the aligned hourly table)
The timing of the window (`hours_start_to_min`, `recovery_hours`, `core_hours`); the lowest Dst and when it happened; the lowest Bz and when, and the hours with Bz at or below -5 nT; the biggest vBs and its mean over the 12 hours before the Dst minimum; the maximum speed, density, dynamic pressure (protons only, 1.67e-6 n v^2) and total field; the Kp and ap maxima; the mean F10.7 and a spike flag. For data quality: the share of field and plasma hours from DSCOVR, the share of plasma hours missing, whether the Dst minimum is final, and whether Kp is preliminary.

## Typical values by class (medians)

| Class | Dst min | Bz min | vBs max | v max | Kp max | F10.7 | Field from DSCOVR | Plasma from DSCOVR |
|---|---|---|---|---|---|---|---|---|
| quiet | -12 | -3.5 | 1.3 | 439 | 2.3 | 121.9 | 100% | 76% |
| moderate | -61 | -8.4 | 4.1 | 563 | 5.3 | 121.3 | 100% | 81% |
| intense | -121 | -16.1 | 8.2 | 557 | 7.3 | 155.1 | 100% | 63% |
| super (2 events) | -369.5 | -40.3 | 27.6 | 879.5 | 8.8 | 216.8 | 91% | 30% |

The southward field (Bz and vBs) separates the storm classes. Speed doesn't separate moderate from intense.

## The 15 evaluation events
They're saved in `evaluation/reference_events/evaluation_set.csv` with each event's window and features.

| Role | Events |
|---|---|
| Headline, super (1) | E216 (2024-05-11, Dst -406) |
| Intense (5) | E158 (2023-04-24, -213), E231 (2024-08-12, -188), E058 (2018-08-26, -176), E027 (2017-05-28, -146), E256 (2025-04-16, -138) |
| Moderate (4) | E039 (2017-09-28, -98), E117 (2022-03-13, -85), E095 (2021-05-12, -61), E002 (2016-08-03, -52) |
| Quiet (5) | E105 (2021-12-05), E121 (2022-05-18), E078 (2020-03-05), E221 (2024-06-12), E143 (2022-11-18) |

For each of these I plan to write a short reference by hand: a few facts the system should state and a few claims it shouldn't make. I haven't written them yet. Caveats: E216's plasma is only 29% from DSCOVR and its window overlaps the next event, and 10 of the 15 events use provisional Dst.

## Event cards (notebook 04)
`data/events/event_cards.jsonl` has one card per event (302). The 15 evaluation cards are also saved as `evaluation/reference_events/cards/E###.json`. A card is a JSON object with these sections: `window`, `dst`, `drivers`, `geomagnetic`, `solar`, `hints`, `data_quality` (source percentages and `caveats`) and `claims_requiring_a_retrieved_source`. A card never says what caused an event or what it affected. That can only come from retrieved documents (decision 29). The `evaluation_role` field is just for my own bookkeeping, and I remove it before a card goes to the model.

The hint rules use thresholds that I chose myself as rules of thumb. Each one can be checked against the table:

| Hint | Rule |
|---|---|
| Kp level | Kp max of at least 5, 6, 7, 8 or 9 gives roughly NOAA G1 to G5 (this is stricter than NOAA for the thirds, so Kp 4.67 counts as below G1) |
| Southward Bz | at least 6 hours at or below -5 nT is "sustained", 1 to 5 hours is "brief", none is "no"; and "strongly southward" if the lowest hourly Bz is -20 nT or less |
| Solar wind speed | maximum hourly speed of 800 km/s or more is very fast, 600 or more is fast, otherwise ordinary |
| Dynamic pressure | a maximum of 10 nPa or more is high (protons only) |
| Electric field | a vBs maximum of 5 mV/m or more is strong |
| Solar activity | mean F10.7 below 100 is low, 100 to 150 is moderate, above 150 is high |
| Timing | hours from Dst first dropping below -50 nT to the minimum, and from the minimum to Dst being back above -30 nT |

Caveats that every card has: the values are hourly means in UTC; the solar wind and field values are shifted to the time they reach Earth (a delay of roughly 25 to 100 minutes); Bz is in GSM coordinates. Caveats that only appear when they apply: DSCOVR supplies under 80% of the plasma or under 90% of the field, Dst is provisional, the window overlaps a neighbouring event (and for 2 events, a deeper Dst value that belongs to the overlapping event), a marginal event, an F10.7 spike, or preliminary Kp.

Checks I ran: all 302 cards agree with a separate recalculation from the aligned table (the Dst minimum over the event's own core, and the Kp, speed, Bz and vBs maxima). The share of cards that carry each hint (quiet / moderate / intense / super): sustained southward Bz 2 / 47 / 100 / 100 %; strong electric field 0 / 25 / 97 / 100 %; high pressure 0 / 18 / 69 / 100 %; strongly southward Bz 0 / 1 / 24 / 100 %; fast or very fast wind 3 / 38 / 45 / 100 %.

## Known limits
- There are only two super storms (May and October 2024), and their plasma values mostly come from OMNI hourly data.
- 14 of the 31 intense and super events have less than half of their plasma hours from DSCOVR.
- Both super storms overlap a neighbouring event, and October 2024 also has only 82% field coverage.
- 17 of the 31 intense and super events pass my `clean` test (no overlap, at least 90% of the field from DSCOVR, not marginal). 9 of those also have at least 75% of the plasma from DSCOVR.
