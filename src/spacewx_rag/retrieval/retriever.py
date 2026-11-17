"""Event-aware retrieval over the Chroma index (notebook 05, decisions 40 and 41)."""

from __future__ import annotations

import pandas as pd

from spacewx_rag.knowledge.embeddings import embed
from spacewx_rag.utils.config import Paths, load_params


def overlaps_event_documents(card: dict) -> bool:
    """True when the event window overlaps the period the event-specific documents cover (May 2024)."""
    a0, b0 = load_params()["knowledge_base"]["event_documents"]["window"]
    a, b = pd.Timestamp(card["window"]["start"][:-1]), pd.Timestamp(card["window"]["end"][:-1])
    return a <= pd.Timestamp(b0) and b >= pd.Timestamp(a0)


def event_queries(card: dict, impacts_by_area: bool = True) -> dict[str, str]:
    """Targeted queries built from a card. `impacts_by_area=False` gives notebook 05's single impact query."""
    g = card["geomagnetic"]["noaa_g_level_approx"]
    cls = card["class"]
    if card["type"] == "storm":
        qs = {
            "physics": f"How southward Bz, fast solar wind and a strong electric field drive a {cls} geomagnetic storm"
        }
        if impacts_by_area:
            for area, tmpl in load_params()["retrieval"]["impact_areas"].items():
                qs[f"impacts: {area}"] = tmpl.format(g=g or "minor")
        else:
            qs["impacts"] = (
                f"Effects of a {g or 'minor'} geomagnetic storm on power systems, spacecraft, satellite navigation and HF radio"
            )
        qs["indices"] = f"What do Kp and Dst values mean for a {cls} storm"
    else:
        qs = {
            "physics": "Solar wind and magnetic field conditions during quiet geomagnetic activity",
            "indices": "What Kp and Dst values indicate quiet conditions and no storm",
        }
    if not card["dst"]["minimum_is_final"]:
        qs["quality"] = "Provisional versus final Dst index values"
    return qs


class Retriever:
    def __init__(self, paths: Paths | None = None):
        import chromadb
        from chromadb.config import Settings

        p = paths or Paths()
        cfg = load_params()
        client = chromadb.PersistentClient(
            path=str(p.vector_store), settings=Settings(anonymized_telemetry=False)
        )
        self.col = client.get_collection(cfg["knowledge_base"]["vector_store"]["collection"])
        self.event_role = cfg["knowledge_base"]["event_documents"]["role"]
        self.top_k = cfg["retrieval"]["top_k"]
        self.impact_top_k = cfg["retrieval"]["impact_top_k"]

    @property
    def general_filter(self) -> dict:
        """Leaves out the event-specific documents."""
        return {"role": {"$ne": self.event_role}}

    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[dict]:
        q = embed([query], query=True)
        res = self.col.query(query_embeddings=q.tolist(), n_results=k, where=where)
        return [
            {
                "chunk_id": i,
                "doc_id": m["doc_id"],
                "title": m["title"],
                "section": m.get("section", ""),
                "source_url": m.get("source_url", ""),
                "licence_status": m.get("licence_status", ""),
                "score": round(1 - d, 3),
                "text": t,
            }
            for i, m, d, t in zip(
                res["ids"][0], res["metadatas"][0], res["distances"][0], res["documents"][0]
            )
        ]

    def retrieve_for_event(
        self, card: dict, k: int | None = None, impacts_by_area: bool = True
    ) -> dict[str, list[dict]]:
        """{query name: results}. Event-specific documents only for events in their window (decision 40)."""
        where = None if overlaps_event_documents(card) else self.general_filter
        out = {}
        for name, q in event_queries(card, impacts_by_area).items():
            kk = k or (self.impact_top_k if name.startswith("impacts: ") else self.top_k)
            out[name] = self.search(q, kk, where)
        return out

    def passages_for_event(self, card: dict) -> list[dict]:
        """Unique passages for the prompt, in query order, each tagged with the queries that found it."""
        seen: dict[str, dict] = {}
        for name, hits in self.retrieve_for_event(card).items():
            for h in hits:
                if h["chunk_id"] in seen:
                    seen[h["chunk_id"]]["queries"].append(name)
                else:
                    seen[h["chunk_id"]] = {**h, "queries": [name]}
        return list(seen.values())
