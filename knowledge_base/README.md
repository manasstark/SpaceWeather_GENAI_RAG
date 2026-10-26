# Knowledge base

These are the reference documents that the retrieval step searches through. I decided to keep the scope small (decision 33): general space weather references plus a few documents about the May 2024 storm, and only public-domain or open-licence sources.

- `raw_docs/` has the original downloads (21 documents, 7.3 MB), retrieved on 2026-10-25.
- `raw_docs/MANIFEST.csv` lists, for each document, its source URL, publisher, format, role (definition, physics, impact or event), size, licence status and notes, and whether it goes into the index by default.
- `processed/` holds the cleaned text and the chunks (made in notebook 05). `vector_store/` holds the index and isn't committed.

By role: 7 definitions (NOAA scales, Kp, Dst, F10.7), 6 physics documents (storms, solar wind, CMEs, coronal holes, the magnetosphere, a Bz tutorial), 6 about impacts (power, satellite communications, satellite drag, HF radio, GNSS, aurora) and 2 about May 2024 (the Hayakawa et al. 2025 paper, CC BY 4.0, and a Wikipedia page, CC BY-SA 4.0).

## Gaps and limits
- I couldn't find an official NOAA report on the May 2024 event, so it rests on one peer-reviewed paper and one secondary summary.
- NOAA has no pages on pipeline or aviation impacts, so questions about those should come back with "no supporting source".
- Document 10 (Kyoto's service home page) isn't much use, so I left it out of the index. Document 11 (the NASA tutorial) is thin.
- Licences: the NOAA pages follow NWS policy (public domain unless noted). GFZ says its index data and graphs are CC BY 4.0, but doesn't say anything about the text. Kyoto's terms (read 2026-10-26) allow scientific use without restrictions with credit, and don't allow commercial use, but they don't say whether the web pages can be redistributed, so documents 10, 20 and 21 are included for academic, non-commercial use with attribution. I haven't verified the licence of the NASA page (11). I need to credit every source.

## Build and retrieval results (notebook 05)
- `processed/texts/` has the cleaned text for each document. `processed/chunks.jsonl` has the 290 chunks with their document, section, source URL and licence. The Chroma index is in `vector_store/` and gets rebuilt when you run notebook 05.
- Embeddings: BAAI/bge-small-en-v1.5, 384 dimensions, cosine similarity.
- On 18 hand-labelled questions I get hit@1 0.78, hit@3 1.0, hit@5 1.0 and MRR 0.889 (the results are saved in `evaluation/results/`). One label was corrected after I read the documents, and the set is small, so it needs a proper review before I quote the numbers.
- The May 2024 documents (12 and 13) are only retrieved for events that overlap 2024-05-08 to 2024-05-20.
