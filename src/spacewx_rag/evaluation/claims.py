"""Claim-by-claim checks of a generated description (docs/architecture.md section 2.6).

Every factual claim is one of:
  summary sentence      -> numbers must be in the card; no cause, no impact, no forecast
  characteristic        -> evidence paths must exist; numbers must be in the card; no cause, no impact
  interpretation/impact -> sources must be retrieved chunks; numbers must be in the card or the
                           cited passages; an LLM judge decides whether the passages support it
A claim is grounded when every check it is subject to passes.
"""

from __future__ import annotations

import re

import pandas as pd

from spacewx_rag.cards import flatten
from spacewx_rag.utils.config import load_params

CAUSE_RE = re.compile(
    r"coronal mass ejection|\bCMEs?\b|high[- ]speed (solar wind )?streams?|coronal holes?|solar flares?",
    re.IGNORECASE,
)
# impact language: the card never contains impacts, so the card-only sections must not either
IMPACT_RE = re.compile(
    r"power grids?|\bgrids?\b|transformers?|blackouts?|satellites?|spacecraft|\bGPS\b|GNSS|navigation|"
    r"HF radio|radio (communication|blackout)|auroras?|auroral",
    re.IGNORECASE,
)
FORECAST_RE = re.compile(
    r"\b(will (occur|happen|cause|affect|be|lead)|forecast(s|ed)?|predict(s|ed|ion)?|expected to)\b",
    re.IGNORECASE,
)
# ISO timestamps / dates and HH:MM times are checked as text, then removed before numbers are read
DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}Z?)?|\b\d{1,2}:\d{2}(?:\s?UTC|Z)?\b")
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_SEP = r"\s*(?:-|–|to|and|through|until)\s*"
_ORD = r"(?:st|nd|rd|th)?"
# written dates, year optional: "May 10, 2024", "August 2 to 3, 2016", "March 13 to March 14, 2022",
# "November 18" (year taken from the event window), and "10-11 May 2024"
WORD_DATE_RE = re.compile(
    rf"\b(?P<m1>{_MON})\s+(?P<d1>\d{{1,2}}){_ORD}(?:{_SEP}(?:(?P<m2>{_MON})\s+)?(?P<d2>\d{{1,2}}){_ORD}\b)?"
    rf"(?:,?\s+(?P<y>\d{{4}}))?\b"
)
WORD_DATE_RE_DMY = re.compile(
    rf"\b(?P<d1>\d{{1,2}}){_ORD}(?:{_SEP}(?P<d2>\d{{1,2}}){_ORD})?\s+(?P<m1>{_MON})(?:\s+(?P<y>\d{{4}}))?\b"
)

# numbers not glued to letters (skips G5, F10.7, d03_c001, R_E ...)
NUM_RE = re.compile(r"(?<![A-Za-z_\d.])[-−–]?\d+(?:\.\d+)?(?![\d_]|[A-Za-z]\d)")


def numbers(text: str) -> list[float]:
    t = DATETIME_RE.sub(" ", WORD_DATE_RE_DMY.sub(" ", WORD_DATE_RE.sub(" ", text)))
    out = []
    for m in NUM_RE.finditer(t):
        s = m.group().replace("−", "-").replace("–", "-")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def card_numbers(card: dict) -> set[float]:
    """Every number that appears in the card: values, numbers inside its text, and in its field names."""
    vals = set()
    for k, v in flatten(card).items():
        if isinstance(v, bool) or v is None:
            pass
        elif isinstance(v, (int, float)):
            vals.add(float(v))
        elif isinstance(v, str):
            vals.update(numbers(v))
        vals.update(abs(x) for x in numbers(k.replace("minus", "-").replace("_", " ")))
    return vals


def card_dates(card: dict) -> set[str]:
    out = set()
    for v in flatten(card).values():
        if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v):
            out.add(v[:10])
            out.add(v[11:16])
    return out


def unmatched_numbers(text: str, allowed: set[float], tol: float | None = None) -> list[float]:
    """Numbers in `text` that match nothing in `allowed` (sign-insensitive, within tolerance)."""
    tol = load_params()["evaluation"]["number_tolerance"] if tol is None else tol
    absd = {abs(a) for a in allowed}
    return [x for x in numbers(text) if not any(abs(abs(x) - a) <= tol for a in absd)]


def word_dates(text: str, years: list[int]) -> list[tuple[str, list[list[pd.Timestamp]]]]:
    """Each written date with the day(s) it names; one candidate list per year when the year is missing."""
    out = []
    t = text
    for rx in (WORD_DATE_RE, WORD_DATE_RE_DMY):
        for m in rx.finditer(t):
            g = m.groupdict()
            m2 = g.get("m2") or g["m1"]
            ys = [int(g["y"])] if g.get("y") else years
            cands = []
            for y in ys:
                days = []
                for mon, d in ((g["m1"], g["d1"]), (m2, g.get("d2"))):
                    if d:
                        try:
                            days.append(pd.Timestamp(f"{y}-{mon[:3]}-{int(d):02d}"))
                        except ValueError:
                            days.append(None)
                cands.append(days)
            out.append((m.group(), cands))
        t = rx.sub(" ", t)
    return out


def unmatched_dates(text: str, card: dict) -> list[str]:
    """Dates that fall outside the event window (written dates) or match no card time (ISO)."""
    ok = card_dates(card)
    if not card["window"].get("start"):  # what-if scenario: any date is invented
        return [
            m.group()
            for rx in (WORD_DATE_RE, WORD_DATE_RE_DMY, DATETIME_RE)
            for m in rx.finditer(text)
        ]
    a = pd.Timestamp(card["window"]["start"][:10])
    b = pd.Timestamp(card["window"]["end"][:10])
    years = sorted({a.year, b.year})
    bad = [
        s
        for s, cands in word_dates(text, years)
        if not any(days and all(d is not None and a <= d <= b for d in days) for days in cands)
    ]
    for m in DATETIME_RE.finditer(text):
        s = m.group()
        if re.match(r"\d{4}-\d{2}-\d{2}", s):
            if s[:10] not in ok:
                bad.append(s)
        elif s[:5].zfill(5) not in ok:
            bad.append(s)
    return bad


def split_sentences(text: str) -> list[str]:
    return [
        s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if len(s.strip()) > 3
    ]


def normalise_ids(ids: list[str]) -> list[str]:
    return [i.strip().strip("[]").strip() for i in ids]


G_HEAD_RE = re.compile(r"\bG\s?([1-5])\b")


def g_number(g: str | None) -> int:
    return int(g[1]) if g else 0


def quote_g_level(quote: str, passages: list[dict]) -> int | None:
    """The NOAA G level whose entry contains `quote` in a scales passage (the last 'G n' heading
    before it), or None when the quote is not under a G heading. Used to stop G5 effects being
    claimed for a weaker storm, which the small judge model let through (decision 56)."""
    q = re.sub(r"\s+", " ", quote).strip().lower()[:60]
    if len(q) < 20:
        return None
    for p in passages:
        text = re.sub(r"\s+", " ", p["text"])
        i = text.lower().find(q)
        if i >= 0:
            heads = list(G_HEAD_RE.finditer(text[:i]))
            return int(heads[-1].group(1)) if heads else None
    return None


def check_output(output: dict, card: dict, passages: list[dict], judge=None) -> list[dict]:
    """One row per claim with the result of every check. `judge(claim, passages) -> (bool, reason)`."""
    allowed = card_numbers(card)
    paths = set(flatten({k: v for k, v in card.items() if k != "evaluation_role"}))
    by_id = {p["chunk_id"]: p for p in passages}
    quiet = card["type"] == "quiet"
    g = card["geomagnetic"]["noaa_g_level_approx"]
    rows = []

    def base(kind, text, **kw):
        bad_num = unmatched_numbers(text, kw.pop("allowed", allowed))
        bad_dates = unmatched_dates(text, card)
        r = {
            "kind": kind,
            "text": text,
            "numbers_ok": not bad_num and not bad_dates,
            "bad_numbers": bad_num + bad_dates,
            "no_forecast": not FORECAST_RE.search(text),
        }
        r.update(kw)
        return r

    for s in split_sentences(output.get("summary", "")):
        rows.append(
            base("summary", s, no_cause=not CAUSE_RE.search(s), no_impact=not IMPACT_RE.search(s))
        )

    for c in output.get("characteristics", []):
        ev = c.get("evidence", [])
        rows.append(
            base(
                "characteristic",
                c["statement"],
                no_cause=not CAUSE_RE.search(c["statement"]),
                no_impact=not IMPACT_RE.search(c["statement"]),
                evidence_ok=bool(ev) and all(e in paths for e in ev),
                evidence=ev,
            )
        )

    for kind, key in (
        ("interpretation", "possible_physical_interpretation"),
        ("impact", "potential_impacts"),
    ):
        for item in output.get(key, []):
            src = normalise_ids(item.get("sources", []))
            cited = [by_id[s] for s in src if s in by_id]
            cited_nums = set()
            for p in cited:
                cited_nums.update(numbers(p["text"]))
            r = base(
                kind,
                item["statement"],
                allowed=allowed | cited_nums,
                sources=src,
                area=item.get("area"),
                sources_ok=bool(src) and len(cited) == len(src),
                fabricated_sources=[s for s in src if s not in by_id],
            )
            if kind == "impact":
                r["quiet_violation"] = quiet
            if judge is not None and cited:
                claim = (
                    item["statement"]
                    if kind == "interpretation"
                    else f"For a geomagnetic storm at about NOAA {g or 'below G1'} level: {item['statement']}"
                )
                verdict = judge(claim, cited)
                r["judge_supported"], r["judge_reason"] = verdict[0], verdict[1]
                quote = verdict[2] if len(verdict) > 2 else ""
                if kind == "impact":
                    lvl = quote_g_level(quote, cited)
                    r["source_g_level"] = lvl
                    r["level_ok"] = lvl is None or lvl <= g_number(g)
            rows.append(r)

    for r in rows:
        checks = [
            r["numbers_ok"],
            r["no_forecast"],
            r.get("no_cause", True),
            r.get("no_impact", True),
            r.get("evidence_ok", True),
            r.get("sources_ok", True),
            not r.get("quiet_violation", False),
        ]
        if r["kind"] in ("interpretation", "impact") and judge is not None:
            checks.append(bool(r.get("judge_supported", False)))
            checks.append(r.get("level_ok", True))
        r["grounded"] = all(checks)
    return rows


def self_contradiction(output: dict) -> bool:
    """True when impacts are given with sources but also listed as having no supporting source."""
    unsup = " ".join(output.get("unsupported_or_unknown", [])).lower()
    return bool(output.get("potential_impacts")) and "impact" in unsup and "no " in unsup


def claim_reason(r: dict) -> str:
    """Plain-language reasons a claim failed its checks ('' when grounded)."""

    def false(k):
        return r.get(k) is False or r.get(k) == "False"

    out = []
    if false("numbers_ok"):
        out.append(f"number not in the card or cited passages {r.get('bad_numbers')}")
    if false("no_cause"):
        out.append("names a cause the card does not give")
    if false("no_impact"):
        out.append("impact claim in a card-only section")
    if false("evidence_ok"):
        out.append("cites a card field that does not exist")
    if false("sources_ok"):
        out.append("no valid source cited")
    if r.get("quiet_violation") is True or r.get("quiet_violation") == "True":
        out.append("impact claimed for a quiet window")
    if false("level_ok"):
        out.append(
            f"effect is from NOAA G{int(float(r.get('source_g_level')))} in the source, higher than this storm"
        )
    if r.get("kind") in ("interpretation", "impact") and false("judge_supported"):
        out.append("judge: the cited passages do not support it")
    if false("no_forecast"):
        out.append("makes a forecast")
    return "; ".join(out)
