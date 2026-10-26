# Spaceweather_GENAI_RAG

**GenAI-Based Multi-Parameter Space Weather Event Analysis and Impact Assessment**
upGrad Masters in ML/AI, GenAI track, Bring Your Own Capstone. Mid-work snapshot, 25 Oct 2026.

> **Where I am right now.** The first five stages are done: getting and cleaning the data, finding the events, building the event cards, building the knowledge base and testing retrieval. They live in notebooks, with the results saved. What is *not* done yet is the part that writes the descriptions with an LLM, the proper `src/` modules, the tests and the Streamlit app. Section 7 has the full table.

## 1. What this project is
Space weather data comes from a few different places: solar wind and the magnetic field measured by the DSCOVR satellite, the Kp and Dst indices that describe how disturbed Earth's magnetic field is, and the F10.7 number for solar activity. To understand what happened during a storm you have to read all of them together, and that takes some expertise.

My idea is to build a system that does this in four steps:

1. Put all the sources on one hourly timeline and automatically find storms and quiet periods.
2. Turn each event into a small structured "event card" (the facts, a few rule-based hints, and warnings about data quality).
3. Look up trustworthy background text in a knowledge base (this is the retrieval part of RAG).
4. Have a language model write a short, structured description of the event that only uses the card and the retrieved text. *(Not built yet.)*

It is not a forecasting model. It only looks at past events and has no live data feed.

## 2. Business objective
- **The problem:** going from raw numbers to a clear assessment of an event takes expert knowledge and a lot of manual work. Dashboards show one parameter at a time. Simple thresholds can tell you that activity is high but not why. A plain LLM will write something that sounds right but may not be supported by the data.
- **Who it's for:** space weather analysts and engineers first, then satellite operators and other space engineering teams.
- **Main success measure:** groundedness, meaning how many of the claims in a generated summary can be backed up by the input data or the retrieved sources. I also plan to look at retrieval relevance, completeness, unsupported claims and consistency.

## 3. Dataset source and licence
Everything about where the data comes from, its formats, what I found while checking it and the caveats is in [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).

| Data | Provider | Licence / terms |
|---|---|---|
| Solar wind and magnetic field, 1-minute (DSCOVR) | NOAA NCEI | US Government data; NWS policy says public domain unless noted |
| Kp, ap, F10.7 | GFZ Potsdam | CC BY 4.0 |
| Kp, ap, F10.7 (daily table) | CelesTrak | free to use, cite CelesTrak |
| Dst, hourly | WDC for Geomagnetism, Kyoto | free for scientific use without restrictions, no commercial use; acknowledge WDC Kyoto and cite the data DOI (doi:10.17593/14515-74000) |
| Hourly cross-check data | NASA OMNIWeb | public; cite King and Papitashvili 2005 |

The knowledge base has 21 documents. Their licence status is listed one by one in `knowledge_base/raw_docs/MANIFEST.csv`. They are 14 NOAA documents, GFZ, Kyoto WDC, a NASA tutorial, the Hayakawa et al. 2025 paper (CC BY 4.0) and a Wikipedia page (CC BY-SA 4.0). Kyoto's terms allow scientific use with credit but don't say whether its web pages can be redistributed, so I included its three documents for academic, non-commercial use with attribution. I haven't verified the licence of the NASA tutorial. The code itself is a course submission, all rights reserved (see `LICENSE`).

**Getting the raw data:** it's about 390 MB so I haven't included it. Run `python scripts/download_data.py`. Use `--dry-run` to just list the files, or `--years 2024` to grab one year as a quick test. The processed files you need to run notebooks 04 and 05 are included.

## 4. Repository structure
| Path | What's in it |
|---|---|
| `notebooks/` | `01_data_audit`, `02_alignment_and_eda`, `03_event_detection`, `04_event_cards`, `05_knowledge_base_and_retrieval` (all done, outputs saved); `06_rag_generation` and `07_evaluation` are empty placeholders |
| `data/processed/` | `aligned_hourly.parquet`: the hourly table, 87,024 rows and 20 columns, from 2016-07-26 to 2026-06-29 (UTC) |
| `data/events/` | `event_table.parquet` / `.csv` (302 events) and `event_cards.jsonl` (302 cards) |
| `data/DATA_SOURCES.md` | where the data comes from, formats, audit findings, licences |
| `data/raw/`, `data/interim/` | created by the download script and notebooks 01-02; **not included** because they're big |
| `knowledge_base/raw_docs/` | the 21 source documents plus `MANIFEST.csv` |
| `knowledge_base/processed/` | cleaned text and `chunks.jsonl` (290 chunks). The Chroma index in `vector_store/` gets rebuilt by notebook 05 |
| `evaluation/reference_events/` | `evaluation_set.csv` (the 15 test events) and their cards |
| `evaluation/results/` | outputs of the retrieval check |
| `docs/` | `architecture.md`, `methodology.md`, `decision_log.md`, `event_definitions.md`, `project_plan.md` |
| `configs/` | `parameters.yaml`, a list of every parameter I used (the code doesn't read it yet) |
| `scripts/` | `download_data.py` |
| `assets/` | figures |
| `src/spacewx_rag/`, `tests/`, `prompts/`, `app/` | only the empty skeleton for now (see the status table) |

## 5. Installation
You need Python 3.10 or newer (I used 3.11).
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m ipykernel install --user --name spaceweather-genai-rag --display-name "Python (Spaceweather_GENAI_RAG)"
cp .env.example .env        # no keys are needed for the notebooks as they are now
```
`requirements.lock.txt` has the exact versions I used. The first time you run notebook 05 it downloads the embedding model (BAAI/bge-small-en-v1.5, about 130 MB) from Hugging Face.

## 6. Execution steps
The notebooks use relative paths, so run them from the `notebooks/` folder with the kernel installed above.

**Quick way (uses the files already included):**
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 04_event_cards.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_knowledge_base_and_retrieval.ipynb
```
You can also open them in JupyterLab (`jupyter lab`) and run all cells. Notebook 04 rebuilds the 302 event cards from the aligned table. Notebook 05 rebuilds the knowledge base text, chunks, embeddings and index, and runs the retrieval checks.

**Full way (rebuilds everything from the raw data):**
```bash
python scripts/download_data.py            # takes roughly an hour
cd notebooks                                # then run 01, 02, 03, 04, 05 in that order
```
Notebooks 01 and 02 each read around 7,300 files and take a few minutes. They save caches in `data/interim/`.

## 7. Current implementation status
| What the brief asks for | Status |
|---|---|
| Business problem and objectives | Done (this README and the proposal) |
| Data collection, preparation, preprocessing | **Done.** Six sources downloaded and checked. Hourly aligned table with a speed-dependent time shift, source labels and quality flags (notebooks 01-02) |
| Finding events and extracting features | **Done.** 202 storms (171 moderate, 29 intense, 2 super) and 100 quiet windows, with features for each (notebook 03) |
| Context preparation | **Done.** 302 event cards with hints, caveats and a list of claims that need a retrieved source (notebook 04) |
| Knowledge base and retrieval | **Done.** 21 documents, 290 chunks, local embeddings, a Chroma index and retrieval that knows about the event. On 18 hand-labelled questions: hit@1 0.78 and hit@3 1.0 (notebook 05) |
| Evaluation set | **Done.** 15 events (1 super, 5 intense, 4 moderate, 5 quiet) with their caveats written down |
| Prompt / interaction design | **Not yet.** The plan is in `docs/architecture.md` |
| Generation with an LLM | **Not yet** |
| Evaluating the generated output | **Not yet.** Only retrieval has been evaluated |
| Modular code and tests | **Not yet.** The logic is still in notebooks and `src/` and `tests/` are empty |
| Streamlit app | **Not yet** (planned for the final submission) |

## 8. Known limitations
- There is no generation step yet, so there is no end-to-end output and no groundedness result.
- The retrieval question set is small (18 questions) and hand-labelled. One label was corrected after I read the document text, so the hit rates are only a rough indication.
- The DSCOVR magnetometer and plasma instruments have long outages (2019-06 to 2020-02, 2025-07 to 2025-10, and early 2026), and the archive ends on 2026-06-29. I fill gaps from OMNI hourly data and every hour records where its value came from.
- The DSCOVR Faraday cup gives wrong speeds in fast solar wind (during May 2024 for at least 34 hours). Because of this, about 35% of the plasma hours come from OMNI hourly values.
- Dst from 2021 onward is provisional, including May 2024, so Kyoto may still revise it. 10 of my 15 evaluation events use provisional Dst.
- Retrieval similarity can't tell supported questions from unsupported ones on related topics (unsupported scored 0.47-0.79, real ones 0.749-0.854). Saying "no supporting source" will need a claim-by-claim check.
- Outside May 2024 the retrieved background is general (by storm class), not specific to the event. I couldn't find an official NOAA report on the May 2024 storm, so that event rests on one paper and one secondary summary.
- Kyoto's terms don't say whether its web pages can be redistributed (I included them for academic, non-commercial use with credit), and the NASA tutorial's licence is unverified.
- The code is still in notebooks, and the configs file is only a reference.

## 9. Planned improvements for the final submission
1. Move the notebook logic into `src/spacewx_rag/` modules with a command-line entry point, configs that the code actually reads, and unit tests.
2. Write the prompt templates (kept in `prompts/`) and a structured output format. Add an LLM generator behind a simple interface, plus an offline baseline generator that doesn't need an API key.
3. Check claims one by one: numbers must match the card, cause and impact claims must point to a retrieved passage, and anything else is reported as "no supporting source".
4. Evaluate on the 15 reference events with hand-written reference points: groundedness (main metric), retrieval relevance, completeness, unsupported claims and consistency. Review and extend the retrieval question set.
5. Build the Streamlit app for browsing events, cards, retrieved sources and generated descriptions.
6. Write the technical report, the presentation and the final README.

## More documentation
[`docs/architecture.md`](docs/architecture.md) (design and planned changes) · [`docs/methodology.md`](docs/methodology.md) · [`docs/decision_log.md`](docs/decision_log.md) (43 decisions with the evidence behind them) · [`docs/event_definitions.md`](docs/event_definitions.md) · [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) · [`knowledge_base/README.md`](knowledge_base/README.md)
