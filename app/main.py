"""Streamlit demo: live NOAA conditions and historical events -> event card -> retrieved sources ->
local LLM description -> claim-by-claim check.

Historical views read the saved outputs, so they work without a model. "Generate" streams the
local LLM (Ollama) as it writes, shows the rule checks at once and the support judge after.

Run from the project folder:  streamlit run app/main.py   (or: make app)
"""

from __future__ import annotations

import html
import os
import re
import threading

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from spacewx_rag.cards import load_cards
from spacewx_rag.evaluation.claims import check_output, claim_reason
from spacewx_rag.generation import load_results
from spacewx_rag.utils.config import Paths, load_env, load_params

load_env()
P = Paths()
st.set_page_config(page_title="Space Weather Event Analyst", page_icon="☀️", layout="wide")

RUN_LABELS = {
    "ollama_rag": "Local LLM + retrieval",
    "ollama_no_retrieval": "Local LLM, no retrieval",
    "baseline": "Offline baseline",
}
CLASS_COLOR = {
    "super": "#c77dff",
    "intense": "#ff4d6d",
    "moderate": "#ff9f1c",
    "active": "#ffd166",
    "quiet": "#2ec4b6",
}
AREA_ICON = {
    "power systems": "⚡",
    "satellites and spacecraft": "🛰️",
    "GNSS navigation": "📡",
    "HF radio": "📻",
    "aurora": "🌌",
    "other": "🔸",
}
KIND_LABEL = {
    "summary": "Summary",
    "characteristic": "Characteristic",
    "interpretation": "Interpretation",
    "impact": "Impact",
}
E = html.escape

# ------------------------------------------------------------------ style

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] {font-family: 'Inter', sans-serif;}
[data-testid="stAppViewContainer"] {background:
  radial-gradient(1200px 500px at 85% -10%, rgba(255,123,0,.10), transparent 60%),
  radial-gradient(900px 500px at -10% 10%, rgba(120,90,255,.10), transparent 60%), #0b1020;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#10163a 0%,#0b1020 100%); border-right:1px solid rgba(255,255,255,.06);}
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: radial-gradient(circle at 12% 25%, rgba(255,160,40,.40), transparent 38%),
       radial-gradient(circle at 95% 0%, rgba(199,125,255,.30), transparent 40%),
       linear-gradient(120deg,#0e1430 0%,#1b1f4b 55%,#2d0f45 100%);
       border:1px solid rgba(255,255,255,.08); border-radius:20px; padding:26px 32px; margin-bottom:14px;}
.hero h1 {font-family:'Space Grotesk',sans-serif; font-size:2.35rem; margin:0; line-height:1.15;
          background:linear-gradient(90deg,#ffe29a,#ff9f1c 45%,#ff4d6d); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#c9cee6; margin:.45rem 0 0 0; font-size:1.02rem;}
.flow {display:flex; gap:.45rem; flex-wrap:wrap; margin-top:14px; align-items:center}
.flow span.s {background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); padding:4px 11px; border-radius:999px; font-size:.8rem; color:#e6e9f5}
.flow span.a {color:#ff9f1c; font-weight:700}
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:8px 0 16px 0}
.kpi {background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.09);
      border-radius:14px; padding:13px 16px; border-top:3px solid var(--accent,#ff7b00)}
.kpi .l {font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:#9aa3c7}
.kpi .v {font-family:'Space Grotesk',sans-serif; font-size:1.65rem; font-weight:700; color:#fff; line-height:1.25}
.kpi .s {font-size:.8rem; color:#c9cee6}
.badge {display:inline-block; padding:3px 13px; border-radius:999px; font-weight:800; font-size:.78rem; letter-spacing:.08em;
        color:#0b1020; background:var(--c); text-transform:uppercase; vertical-align:middle}
.ev-title {font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; margin:.2rem 0 .1rem 0}
.muted {color:#9aa3c7; font-size:.88rem}
.card {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:13px 16px; margin-bottom:10px}
.card h4 {margin:0 0 .35rem 0; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; color:#9aa3c7}
.summary {border-left:4px solid #ff9f1c; background:linear-gradient(90deg,rgba(255,159,28,.13),rgba(255,159,28,.03));
          padding:15px 19px; border-radius:12px; font-size:1.06rem; line-height:1.6; margin-bottom:14px}
.chip {display:inline-block; background:rgba(120,140,255,.14); color:#b3bdff; border:1px solid rgba(120,140,255,.3);
       padding:0 7px; border-radius:6px; font-family:ui-monospace,monospace; font-size:.72rem; margin:3px 4px 0 0}
.impact {display:flex; gap:12px; align-items:flex-start}
.impact .ico {font-size:1.55rem; line-height:1.2}
.impact .ar {font-weight:700; font-size:.85rem; color:#ffd166}
.claim {display:flex; gap:12px; align-items:flex-start; padding:9px 14px; border-radius:12px; margin-bottom:7px;
        background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07)}
.claim.ok {border-left:4px solid #2ec4b6}
.claim.bad {border-left:4px solid #ff4d6d; background:rgba(255,77,109,.07)}
.claim.wait {border-left:4px solid #ffd166}
.claim .ic {font-size:1.15rem}
.claim .t {font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; color:#9aa3c7}
.claim .why {color:#ff8fa3; font-size:.84rem; margin-top:3px}
.score {font-family:'Space Grotesk',sans-serif; font-size:2.4rem; font-weight:700; line-height:1}
.status {padding:10px 14px; border-radius:12px; font-size:.9rem; margin-bottom:10px}
.status.ok {background:rgba(46,196,182,.12); border:1px solid rgba(46,196,182,.45)}
.status.warn {background:rgba(255,209,102,.12); border:1px solid rgba(255,209,102,.45)}
.steps {display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:8px}
.step {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px 16px}
.step .n {font-family:'Space Grotesk',sans-serif; color:#ff9f1c; font-weight:700; font-size:1.3rem}
.step b {display:block; margin:.2rem 0}
.stTabs [data-baseweb="tab-list"] {gap:6px}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:1rem; background:rgba(255,255,255,.04); border-radius:10px 10px 0 0; padding:8px 16px}
.stTabs [aria-selected="true"] {background:rgba(255,123,0,.15)}
div.stButton > button {border-radius:10px; font-weight:600}

.ecard-head {display:flex; align-items:center; gap:12px; margin:2px 0 12px 0}
.ecards {display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px}
.ecard {background:linear-gradient(160deg,rgba(255,255,255,.065),rgba(255,255,255,.015)); border:1px solid rgba(255,255,255,.09);
        border-radius:16px; padding:14px 16px 8px 16px; box-shadow:0 6px 24px rgba(0,0,0,.25)}
.ecard .hd {display:flex; align-items:center; gap:10px; font-weight:700; font-size:.95rem; color:#fff; margin-bottom:6px}
.ecard .hd .ic {width:32px; height:32px; border-radius:10px; display:grid; place-items:center; font-size:1.05rem;
                background:linear-gradient(135deg,rgba(255,159,28,.25),rgba(199,125,255,.2))}
.kv {display:flex; justify-content:space-between; align-items:center; gap:14px; padding:7px 2px;
     border-top:1px dashed rgba(255,255,255,.08); font-size:.88rem}
.ecard .hd + .kv {border-top:none}
.kv .k {color:#9aa3c7}
.kv .v {font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:.98rem; text-align:right; white-space:nowrap; color:#fff}
.kv .u {color:#9aa3c7; font-family:'Inter',sans-serif; font-weight:500; font-size:.75rem; margin-left:4px}
.pill {display:inline-block; padding:2px 11px; border-radius:999px; font-family:'Inter',sans-serif; font-size:.72rem;
       font-weight:800; letter-spacing:.05em; text-transform:uppercase; color:#0b1020}
.bar {display:inline-block; width:96px; height:7px; background:rgba(255,255,255,.09); border-radius:99px; overflow:hidden;
      vertical-align:middle; margin-right:9px}
.bar > i {display:block; height:100%; border-radius:99px}
.lock {display:flex; gap:9px; padding:7px 2px; border-top:1px dashed rgba(255,255,255,.08); font-size:.86rem; color:#e6e9f5}

</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ data and model


@st.cache_data
def cards() -> dict:
    return load_cards()


@st.cache_data
def eval_set() -> pd.DataFrame:
    return pd.read_csv(P.evaluation_set)


@st.cache_data
def aligned() -> pd.DataFrame:
    return pd.read_parquet(P.aligned)


@st.cache_data
def results(run: str) -> dict:
    return {(r["event_id"], r["run"]): r for r in load_results(run)}


@st.cache_data
def claims() -> pd.DataFrame:
    return pd.read_csv(P.results / "claims.csv")


@st.cache_data
def metrics() -> pd.DataFrame:
    return pd.read_csv(P.results / "generation_metrics.csv")


@st.cache_data
def manifest() -> pd.DataFrame:
    return pd.read_csv(P.manifest).set_index("doc_id")


@st.cache_resource
def retriever():
    from spacewx_rag.retrieval import Retriever

    return Retriever()


@st.cache_resource
def generator():
    from spacewx_rag.generation import OllamaGenerator

    return OllamaGenerator()


def ollama_up() -> bool:
    try:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return requests.get(f"{host}/api/version", timeout=1).ok
    except requests.RequestException:
        return False


@st.cache_resource
def warm_model() -> bool:
    """Load the LLM in the background once per server start, so the first click is fast."""
    if ollama_up():
        threading.Thread(target=lambda: generator().warm_up(), daemon=True).start()
        return True
    return False


# ------------------------------------------------------------------ rendering


def kpi(label: str, value: str, sub: str = "", accent: str = "#ff7b00") -> str:
    return (
        f'<div class="kpi" style="--accent:{accent}"><div class="l">{E(label)}</div>'
        f'<div class="v">{E(value)}</div><div class="s">{E(sub)}</div></div>'
    )


def badge(cls: str) -> str:
    return f'<span class="badge" style="--c:{CLASS_COLOR.get(cls, "#9aa3c7")}">{E(cls)}</span>'


def show_kpis(card: dict) -> None:
    d, g, dr, s = card["dst"], card["geomagnetic"], card["drivers"], card["solar"]
    dst_acc = "#ff4d6d" if d["min_nT"] <= -100 else "#ff9f1c" if d["min_nT"] <= -50 else "#2ec4b6"
    kp_acc = "#ff4d6d" if g["kp_max"] >= 7 else "#ff9f1c" if g["kp_max"] >= 5 else "#2ec4b6"
    bz_acc = (
        "#ff4d6d"
        if dr["bz_min_hourly_nT"] <= -20
        else "#ff9f1c"
        if dr["bz_min_hourly_nT"] <= -5
        else "#2ec4b6"
    )
    v_acc = (
        "#ff4d6d"
        if dr["speed_max_km_per_s"] >= 800
        else "#ff9f1c"
        if dr["speed_max_km_per_s"] >= 600
        else "#2ec4b6"
    )
    st.markdown(
        '<div class="kpis">'
        + kpi(
            "Dst minimum",
            f"{d['min_nT']} nT",
            "final" if d["minimum_is_final"] else "provisional / real-time",
            dst_acc,
        )
        + kpi(
            "Kp maximum",
            f"{g['kp_max']:g}",
            f"NOAA {g['noaa_g_level_approx']}" if g["noaa_g_level_approx"] else "below G1",
            kp_acc,
        )
        + kpi(
            "Lowest hourly Bz",
            f"{dr['bz_min_hourly_nT']} nT",
            f"{dr['hours_bz_at_or_below_minus5']} h at or below -5 nT",
            bz_acc,
        )
        + kpi(
            "Max solar wind",
            f"{dr['speed_max_km_per_s']} km/s",
            f"pressure up to {dr['dynamic_pressure_max_nPa_protons_only']} nPa",
            v_acc,
        )
        + kpi("Solar activity", f"{s['f107_mean_sfu']} sfu", f"F10.7 · {s['level']}", "#c77dff")
        + "</div>",
        unsafe_allow_html=True,
    )


def show_card_details(card: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="card"><h4>Rule-based hints (computed from the data)</h4>'
            + "".join(f"<div>• {E(h)}</div>" for h in card["hints"])
            + "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card"><h4>Data-quality caveats</h4>'
            + "".join(f"<div>• {E(h)}</div>" for h in card["data_quality"]["caveats"])
            + "</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "The card never states a cause or an impact. Those may only come from retrieved documents."
    )
    with st.expander("🧾 Full event card (everything the model receives)"):
        st.markdown(card_table_html(card), unsafe_allow_html=True)
        if st.toggle("Show the raw JSON", key=f"raw_{card['event_id']}"):
            st.json(card)


def _fmt(v, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str) and len(v) == 17 and v[10] == "T" and v.endswith("Z"):
        return f"{v[:10]} {v[11:16]} UTC"
    return f"{v:g} {unit}".strip() if isinstance(v, (int, float)) else f"{v} {unit}".strip()


CARD_LABELS = {
    "🕒 Time window": [
        ("window.start", "Start", ""),
        ("window.end", "End", ""),
        ("window.hours", "Length", "h"),
    ],
    "🌍 Dst (ring current)": [
        ("dst.min_nT", "Dst minimum", "nT"),
        ("dst.min_time", "Time of minimum", ""),
        ("dst.first_below_minus50_time", "First below −50 nT", ""),
        ("dst.hours_from_minimum_to_back_above_minus30", "Recovery to −30 nT", "h"),
        ("dst.minimum_is_final", "Final (not provisional)", ""),
    ],
    "🌬️ Solar wind drivers": [
        ("drivers.bz_min_hourly_nT", "Lowest hourly Bz", "nT"),
        ("drivers.bz_min_time", "Time of lowest Bz", ""),
        ("drivers.hours_bz_at_or_below_minus5", "Hours with Bz ≤ −5 nT", "h"),
        ("drivers.vbs_max_mV_per_m", "Max electric field (vBs)", "mV/m"),
        ("drivers.vbs_mean_12h_before_dst_min_mV_per_m", "Mean vBs, 12 h before Dst min", "mV/m"),
        ("drivers.speed_max_km_per_s", "Max speed", "km/s"),
        ("drivers.density_max_per_cm3", "Max density", "/cm³"),
        ("drivers.dynamic_pressure_max_nPa_protons_only", "Max dynamic pressure", "nPa"),
        ("drivers.total_field_max_nT", "Max total field |B|", "nT"),
    ],
    "🧲 Geomagnetic indices": [
        ("geomagnetic.kp_max", "Kp maximum", ""),
        ("geomagnetic.noaa_g_level_approx", "NOAA storm level (approx.)", ""),
        ("geomagnetic.ap_max", "ap maximum", ""),
    ],
    "☀️ Solar activity": [
        ("solar.f107_mean_sfu", "Mean F10.7", "sfu"),
        ("solar.level", "Level", ""),
    ],
}


def card_table_html(card: dict) -> str:
    """The event card as styled, grouped key/value panels instead of raw JSON."""

    def get(path):
        v = card
        for k in path.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        return v

    def sev(path, v):
        """Accent colour for values that signal disturbance."""
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        rules = {
            "dst.min_nT": (-100, -50, "le"),
            "drivers.bz_min_hourly_nT": (-20, -5, "le"),
            "drivers.speed_max_km_per_s": (800, 600, "ge"),
            "geomagnetic.kp_max": (7, 5, "ge"),
            "drivers.dynamic_pressure_max_nPa_protons_only": (20, 10, "ge"),
            "drivers.vbs_max_mV_per_m": (10, 5, "ge"),
        }
        if path not in rules:
            return None
        hi, mid, op = rules[path]
        hit = (lambda x, t: x <= t) if op == "le" else (lambda x, t: x >= t)
        return "#ff4d6d" if hit(v, hi) else "#ff9f1c" if hit(v, mid) else "#2ec4b6"

    def pill(text, color):
        return f'<span class="pill" style="background:{color}">{E(text)}</span>'

    def value(path, v, unit):
        if path == "geomagnetic.noaa_g_level_approx":
            colors = {
                "G1": "#ffd166",
                "G2": "#ff9f1c",
                "G3": "#ff7b00",
                "G4": "#ff4d6d",
                "G5": "#c77dff",
            }
            return pill(v, colors[v]) if v else pill("below G1", "#5a6390")
        if path == "dst.minimum_is_final":
            return pill("final", "#2ec4b6") if v else pill("provisional", "#ffd166")
        if path == "solar.level":
            return pill(
                v, {"low": "#2ec4b6", "moderate": "#ffd166", "high": "#ff9f1c"}.get(v, "#9aa3c7")
            )
        if path.endswith("_pct") and isinstance(v, (int, float)):
            bad = path.endswith("missing_pct")
            c = (
                ("#ff4d6d" if v > 10 else "#2ec4b6")
                if bad
                else ("#2ec4b6" if v >= 90 else "#ffd166" if v >= 50 else "#ff4d6d")
            )
            return f'<span class="bar"><i style="width:{max(min(v, 100), 2)}%;background:{c}"></i></span>{v:g}<span class="u">%</span>'
        txt = _fmt(v)
        if txt == "–":
            return '<span class="muted">–</span>'
        color = sev(path, v)
        style = f' style="color:{color}"' if color else ""
        return f"<span{style}>{E(txt)}</span>" + (
            f'<span class="u">{E(unit)}</span>' if unit else ""
        )

    dq = card["data_quality"]
    ds_labels = {
        "field_from_dscovr_pct": "Field from DSCOVR",
        "plasma_from_dscovr_pct": "Plasma from DSCOVR",
        "plasma_missing_pct": "Plasma missing",
        "solar_wind_monitor": "Solar wind monitor",
        "field_coverage_pct": "Field coverage",
        "plasma_coverage_pct": "Plasma coverage",
    }
    groups = dict(CARD_LABELS)
    groups["📡 Data sources"] = [
        (f"data_quality.{k}", ds_labels.get(k, k.replace("_", " ")), "")
        for k in dq
        if k != "caveats"
    ]
    panels = []
    for title, rows in groups.items():
        icon, name = title.split(" ", 1)
        body = "".join(
            f'<div class="kv"><span class="k">{E(lab)}</span><span class="v">{value(path, get(path), unit)}</span></div>'
            for path, lab, unit in rows
        )
        panels.append(
            f'<div class="ecard"><div class="hd"><span class="ic">{icon}</span>{E(name)}</div>{body}</div>'
        )
    locks = "".join(
        f'<div class="lock"><span>🔒</span><span>{E(x)}</span></div>'
        for x in card.get("claims_requiring_a_retrieved_source", [])
    )
    panels.append(
        f'<div class="ecard"><div class="hd"><span class="ic">🔒</span>Only allowed with a cited source</div>'
        f'<div class="muted" style="font-size:.8rem;margin-bottom:4px">The model may not state these from the '
        f"card alone</div>{locks}</div>"
    )
    head = (
        f'<div class="ecard-head">{badge(card["class"])}<span style="font-weight:700">{E(card["event_id"])}</span>'
        f'<span class="muted">type: {E(card["type"])} · hints and caveats are shown above</span></div>'
    )
    return head + '<div class="ecards">' + "".join(panels) + "</div>"


def storm_plot(d: pd.DataFrame, start: str, end: str, height: int = 600) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=(
            "Dst (nT): ring current strength",
            "Bz GSM (nT): southward is negative",
            "Solar wind speed (km/s)",
            "Kp (planetary index)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["dst"],
            name="Dst",
            line={"color": "#c77dff", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(199,125,255,.12)",
        ),
        1,
        1,
    )
    for y, lab in ((-50, "storm"), (-100, "intense")):
        if d["dst"].min() < y + 40:
            fig.add_hline(
                y=y,
                line={"dash": "dot", "color": "#9aa3c7", "width": 1},
                row=1,
                col=1,
                annotation_text=lab,
                annotation_position="bottom right",
                annotation_font_color="#9aa3c7",
            )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["bz"], name="Bz", line={"color": "#ff4d6d", "width": 2}), 2, 1
    )
    fig.add_hline(y=0, line={"color": "#5a6390", "width": 1}, row=2, col=1)
    omni = d["plasma_src"] == "omni"
    fig.add_trace(
        go.Scatter(
            x=d.index, y=d["v"].where(~omni), name="speed", line={"color": "#4cc9f0", "width": 2}
        ),
        3,
        1,
    )
    if omni.any():
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d["v"].where(omni),
                name="speed (OMNI fallback)",
                line={"color": "#ff9f1c", "dash": "dot", "width": 2},
            ),
            3,
            1,
        )
    kp_col = [
        "#ff4d6d" if k >= 7 else "#ff9f1c" if k >= 5 else "#2ec4b6" for k in d["kp"].fillna(0)
    ]
    fig.add_trace(go.Bar(x=d.index, y=d["kp"], name="Kp", marker_color=kp_col), 4, 1)
    fig.add_hline(
        y=5,
        line={"dash": "dot", "color": "#9aa3c7", "width": 1},
        row=4,
        col=1,
        annotation_text="G1 storm level",
        annotation_position="top left",
        annotation_font_color="#9aa3c7",
    )
    fig.add_vrect(x0=start, x1=end, fillcolor="#ffd166", opacity=0.08, line_width=0)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        legend={"orientation": "h", "y": -0.07},
        bargap=0,
        font={"family": "Inter"},
    )
    fig.update_annotations(font_size=12)
    return fig


def _dark(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        legend={"orientation": "h", "y": -0.1},
        bargap=0,
        font={"family": "Inter"},
    )
    fig.update_annotations(font_size=12)
    return fig


def live_solar_wind_plot(snap: dict) -> go.Figure:
    """1-minute solar wind as measured at L1 (about an hour upstream of Earth)."""
    mag = pd.DataFrame(snap["mag"])
    wind = pd.DataFrame(snap["wind"])
    for d in (mag, wind):
        d["t"] = pd.to_datetime(d["time_tag"]).dt.tz_localize(None)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Magnetic field (nT): Bz southward is negative, Bt total",
            "Solar wind speed (km/s)",
            "Proton density (/cm³)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=mag["t"], y=mag["bt"], name="Bt", line={"color": "#9aa3c7", "width": 1.2}),
        1,
        1,
    )
    fig.add_trace(
        go.Scatter(x=mag["t"], y=mag["bz_gsm"], name="Bz", line={"color": "#ff4d6d", "width": 1.4}),
        1,
        1,
    )
    fig.add_hline(y=0, line={"color": "#5a6390", "width": 1}, row=1, col=1)
    fig.add_hline(
        y=-5,
        line={"dash": "dot", "color": "#ff9f1c", "width": 1},
        row=1,
        col=1,
        annotation_text="−5 nT",
        annotation_position="bottom left",
        annotation_font_color="#ff9f1c",
    )
    fig.add_trace(
        go.Scatter(
            x=wind["t"],
            y=wind["proton_speed"],
            name="speed",
            line={"color": "#4cc9f0", "width": 1.4},
        ),
        2,
        1,
    )
    fig.add_trace(
        go.Scatter(
            x=wind["t"],
            y=wind["proton_density"],
            name="density",
            line={"color": "#ffd166", "width": 1.2},
        ),
        3,
        1,
    )
    return _dark(fig, 520)


def geomag_plot(d: pd.DataFrame, start: str, end: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Dst (nT): ring current strength", "Kp (planetary index)"),
    )
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["dst"],
            name="Dst",
            line={"color": "#c77dff", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(199,125,255,.12)",
        ),
        1,
        1,
    )
    kp_col = [
        "#ff4d6d" if k >= 7 else "#ff9f1c" if k >= 5 else "#2ec4b6" for k in d["kp"].fillna(0)
    ]
    fig.add_trace(go.Bar(x=d.index, y=d["kp"], name="Kp", marker_color=kp_col), 2, 1)
    fig.add_hline(
        y=5,
        line={"dash": "dot", "color": "#9aa3c7", "width": 1},
        row=2,
        col=1,
        annotation_text="G1 storm level",
        annotation_position="top left",
        annotation_font_color="#9aa3c7",
    )
    fig.update_yaxes(range=[0, 9], row=2, col=1)
    fig.add_vrect(x0=start, x1=end, fillcolor="#ffd166", opacity=0.08, line_width=0)
    return _dark(fig, 380)


def chips(ids) -> str:
    return (
        "".join(f'<span class="chip">{E(i)}</span>' for i in ids)
        or '<span class="muted">no source</span>'
    )


def show_description(out: dict) -> None:
    st.markdown(f'<div class="summary">{E(out["summary"])}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown(
            '<div class="card"><h4>Characteristics · from the event card</h4>'
            + "".join(
                f'<div style="margin-bottom:6px">• {E(c["statement"])}<br>{chips(c["evidence"])}</div>'
                for c in out["characteristics"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        if out["possible_physical_interpretation"]:
            st.markdown(
                '<div class="card"><h4>Possible physical interpretation · from the sources</h4>'
                + "".join(
                    f'<div style="margin-bottom:6px">🧲 {E(i["statement"])}<br>{chips(i["sources"])}</div>'
                    for i in out["possible_physical_interpretation"]
                )
                + "</div>",
                unsafe_allow_html=True,
            )
    with c2:
        if out["potential_impacts"]:
            for i in out["potential_impacts"]:
                st.markdown(
                    f'<div class="card impact"><div class="ico">{AREA_ICON.get(i["area"], "🔸")}</div><div>'
                    f'<div class="ar">{E(i["area"])}</div>{E(i["statement"])}<br>{chips(i["sources"])}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="card impact"><div class="ico">🟢</div><div><div class="ar">Potential impacts</div>'
                "None stated.</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="card"><h4>Caveats</h4>'
            + (
                "".join(f"<div>⚠️ {E(x)}</div>" for x in out["caveats"])
                or '<span class="muted">none</span>'
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        if out["unsupported_or_unknown"]:
            st.markdown(
                '<div class="card"><h4>Unsupported or unknown</h4>'
                + "".join(f"<div>❔ {E(x)}</div>" for x in out["unsupported_or_unknown"])
                + "</div>",
                unsafe_allow_html=True,
            )


def _bool(v) -> bool | None:
    if isinstance(v, str):
        return {"True": True, "False": False}.get(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return bool(v)


def show_claims(rows: list[dict], pending: bool = False) -> None:
    """Claim cards. With `pending`, cause and impact claims that pass the rules wait for the judge."""
    if not rows:
        st.info("No claim check available for this output.")
        return
    html_rows, done, good = [], 0, 0
    for r in rows:
        ok = _bool(r.get("grounded"))
        waiting = pending and r["kind"] in ("interpretation", "impact") and ok
        if waiting:
            cls, ic, why = "wait", "⏳", "checking the cited sources…"
        elif ok:
            cls, ic, why = "ok", "✅", ""
            done += 1
            good += 1
        else:
            cls, ic, why = "bad", "❌", claim_reason(r)
            done += 1
        src = r.get("sources")
        if isinstance(src, str) and src.startswith("["):
            src = [s.strip(" '\"") for s in src.strip("[]").split(",") if s.strip()]
        elif not isinstance(src, list):
            src = []  # summary and characteristic rows have no sources (NaN in the CSV)
        html_rows.append(
            f'<div class="claim {cls}"><div class="ic">{ic}</div><div style="flex:1">'
            f'<div class="t">{KIND_LABEL.get(r["kind"], r["kind"])}</div>{E(str(r["text"]))}'
            + (f"<br>{chips(src)}" if src else "")
            + (f'<div class="why">{E(why)}</div>' if why else "")
            + "</div></div>"
        )
    n = len(rows)
    pct = good / done if done else 0
    color = "#2ec4b6" if pct >= 0.9 else "#ffd166" if pct >= 0.75 else "#ff4d6d"
    sub = f"of {done} checked so far" if pending else f"{pct:.0%} of {n} claims"
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(
            f'<div class="card" style="text-align:center"><h4>Grounded</h4>'
            f'<div class="score" style="color:{color}">{good}/{done}</div><div class="muted">{sub}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("".join(html_rows), unsafe_allow_html=True)


def show_sources(passages: list[dict], cited: set[str]) -> None:
    if not passages:
        st.info("This run used no retrieved passages (card only).")
        return
    man = manifest()
    st.caption(f"{len(passages)} passages from targeted queries · 📌 = cited in the description")
    for p in passages:
        mark = " 📌" if p["chunk_id"] in cited else ""
        with st.expander(
            f"{p['chunk_id']} · {p['title'][:70]} · similarity {p['score']:.2f}{mark}"
        ):
            st.write(p["text"])
            st.caption(f"Found by: {', '.join(p.get('queries') or [])}")
            if p.get("doc_id") in man.index:
                row = man.loc[p["doc_id"]]
                st.caption(
                    f"{row['publisher']} · {row['source_url']}  \nLicence: {str(row['licence_status'])[:240]}"
                )


TERM_STYLE = (
    "background:#000;color:#e5e5e5;font-family:'JetBrains Mono',Menlo,Monaco,Consolas,monospace;"
    "font-size:13.5px;line-height:1.6;padding:14px 16px;border-radius:6px;border:1px solid #333;"
    "white-space:pre-wrap;word-break:break-word;max-height:480px;overflow-y:auto;margin:0"
)
USER_PS1 = '<span style="color:#4ec94e;font-weight:600">user@spacewx</span>:<span style="color:#569cd6">~</span>$ '
AI_PS1 = '<span style="color:#ff9f1c;font-weight:600">genai@spacewx</span>:<span style="color:#569cd6">~</span>$ '
CURSOR = '<span style="background:#e5e5e5;animation:tb 1s steps(1) infinite">&nbsp;</span>'


def chat_terminal(
    lines: list[tuple[str, str]], typing: str | None = None, done: bool = False
) -> str:
    """A terminal chat: every line has its own prompt; `typing` is the line being written now."""
    rows = [(USER_PS1 if who == "user" else AI_PS1) + E(text) for who, text in lines[-40:]]
    if typing is not None:
        rows.append(AI_PS1 + E(typing) + CURSOR)
    elif done:
        rows.append(USER_PS1 + CURSOR)
    return (
        "<style>@keyframes tb{50%{opacity:0}}</style>"
        + f'<pre style="{TERM_STYLE}">'
        + "\n".join(rows)
        + "</pre>"
    )


_ITEM = re.compile(
    r'(?:"area"\s*:\s*"(?P<area>[^"]*)"\s*,\s*)?"(?P<key>summary|statement)"\s*:\s*"'
    r'(?P<text>(?:[^"\\]|\\.)*)(?P<end>"?)'
)


def stream_items(buf: str) -> tuple[list[str], str | None]:
    """Readable lines from the model's (unfinished) JSON: finished summary sentences and statements,
    plus the one still being written. Impacts get their area as a prefix. No JSON is shown."""
    import json

    def text_of(raw: str) -> str:
        try:
            return json.loads('"' + raw.rstrip("\\") + '"')
        except json.JSONDecodeError:
            return raw

    done, typing = [], None
    for m in _ITEM.finditer(buf):
        txt = text_of(m["text"])
        prefix = f"[{m['area']}] " if m["area"] else ""
        if m["key"] == "summary":
            parts = [x for x in re.split(r"(?<=[.!?])\s+", txt) if x]
            if m["end"]:
                done += parts
            else:
                done += parts[:-1]
                typing = parts[-1] if parts else ""
        elif m["end"]:
            done.append(prefix + txt)
        else:
            typing = prefix + txt
    return done, typing


def output_lines(out: dict) -> list[str]:
    """The finished answer as terminal lines (same order as streamed)."""
    import re

    lines = [x for x in re.split(r"(?<=[.!?])\s+", out["summary"]) if x]
    lines += [c["statement"] for c in out["characteristics"]]
    lines += [i["statement"] for i in out["possible_physical_interpretation"]]
    lines += [f"[{i['area']}] {i['statement']}" for i in out["potential_impacts"]]
    return lines


def generate_flow(card: dict, key: str, question: str) -> None:
    """Chat-style terminal: the question, retrieval, then the answer typed line by line as the model
    writes it (parsed from the JSON stream), then the rule checks (instant)."""
    import time

    term = st.empty()
    convo = [("user", question)]
    term.html(chat_terminal(convo, typing="searching 21 reference documents..."))
    passages = retriever().passages_for_event(card)
    docs = len({p["doc_id"] for p in passages})
    convo.append(
        (
            "ai",
            f"searching 21 reference documents... {len(passages)} passages found in {docs} of them",
        )
    )
    term.html(chat_terminal(convo, typing="thinking..."))
    buf, count, t0 = [], [0], time.time()

    def on_token(piece: str) -> None:
        buf.append(piece)
        count[0] += 1
        if count[0] % 2 == 0:
            done, typing = stream_items("".join(buf))
            term.html(
                chat_terminal(
                    convo + [("ai", x) for x in done],
                    typing=typing if typing is not None else "thinking...",
                )
            )

    res = generator().generate(card, passages, on_token=on_token)
    if not res.valid:
        convo.append(("ai", f"sorry, my answer came out malformed ({res.error}). try again."))
        term.html(chat_terminal(convo, done=True))
        st.session_state[key] = {"error": res.error, "raw": res.raw}
        return
    convo += [("ai", x) for x in output_lines(res.output)]
    convo.append(
        (
            "ai",
            f"done in {time.time() - t0:.0f} s. now checking every claim against the data and the sources...",
        )
    )
    final_html = chat_terminal(convo, done=True)
    term.html(final_html)
    rule_rows = check_output(res.output, card, passages, judge=None)
    st.session_state[key] = {
        "output": res.output,
        "passages": passages,
        "claims": rule_rows,
        "pending": True,
        "seconds": res.seconds,
        "card": card,
        "terminal": final_html,
    }


def judge_pending(key: str) -> None:
    """Second step: the support judge on the cause and impact claims, with a progress bar."""
    from spacewx_rag.evaluation.judge import SupportJudge

    s = st.session_state[key]
    targets = sum(r["kind"] in ("interpretation", "impact") for r in s["claims"])
    bar = st.progress(0.0, text="🧑‍⚖️ Verifying the cited sources…")
    j = SupportJudge(generator(), cache_path=P.root / "data" / "live" / "judge_cache.jsonl")
    n = [0]

    def judge(claim, passages):
        out = j(claim, passages)
        n[0] += 1
        bar.progress(
            min(n[0] / max(targets, 1), 1.0),
            text=f"🧑‍⚖️ Verifying the cited sources… {n[0]}/{targets}",
        )
        return out

    s["claims"] = check_output(s["output"], s["card"], s["passages"], judge)
    s["pending"] = False
    bar.empty()


def show_generated(key: str) -> None:
    s = st.session_state.get(key)
    if not s:
        return
    if "error" in s:
        st.error(f"Invalid JSON from the model: {s['error']}")
        st.code(s["raw"][-1200:])
        return
    if s.get("terminal"):
        with st.expander("🖥️ Terminal conversation"):
            st.html(s["terminal"])
    show_description(s["output"])
    st.markdown("#### ✅ Claim-by-claim check")
    if s["pending"]:
        show_claims(s["claims"], pending=True)
        judge_pending(key)
        st.rerun()
    show_claims(s["claims"])
    with st.expander("📚 Retrieved sources"):
        cited = {
            x
            for k in ("possible_physical_interpretation", "potential_impacts")
            for i in s["output"][k]
            for x in i["sources"]
        }
        show_sources(s["passages"], cited)


# ------------------------------------------------------------------ layout

up = ollama_up()
warm_model()

st.markdown(
    """
<div class="hero">
  <h1>☀️ Space Weather Event Analyst</h1>
  <p>Reads solar wind, geomagnetic indices and solar activity together, retrieves trusted references, and writes a
  grounded explanation with a local AI model. Every claim is then checked.</p>
  <div class="flow"><span class="s">🛰️ NOAA / DSCOVR data</span><span class="a">→</span><span class="s">🧮 event card</span>
  <span class="a">→</span><span class="s">📚 retrieval (21 documents)</span><span class="a">→</span><span class="s">🤖 local LLM</span>
  <span class="a">→</span><span class="s">✅ claim-by-claim check</span></div>
</div>""",
    unsafe_allow_html=True,
)

ev, cs = eval_set(), cards()
with st.sidebar:
    st.markdown("## ☀️ Event Analyst")
    st.markdown(
        f"**Local LLM** {'🟢 ready' if up else '⚪ offline'}  \n`{load_params()['generation']['model']}` via Ollama"
    )
    if not up:
        st.caption("Start it with `brew services start ollama` to enable generation.")
    st.divider()
    st.markdown("### 📚 Historical event")
    labels = {
        r.event_id: f"{r.event_id} · {r.cls} · {str(r.min_time)[:10]} · Dst {r.min_dst}"
        for r in ev.itertuples()
    }
    eid = st.selectbox(
        "Evaluation event",
        list(labels),
        format_func=labels.get,
        index=0,
        label_visibility="collapsed",
    )
    run = st.radio("Show the description from", list(RUN_LABELS), format_func=RUN_LABELS.get)
    st.divider()
    st.markdown("### 🎬 Demo path")
    st.markdown(
        "1. **🛰️ Live now**: fetch NOAA, then generate  \n"
        "1b. **🎛️ What-if**: pick a preset, slide, ask the AI  \n"
        "2. **E216**: the May 2024 super storm  \n"
        "3. **E143**: a quiet day, no invented drama  \n"
        "4. **E231**: the checker catches G5 effects claimed for a G4 storm  \n"
        "5. **📊 Results**"
    )

tab_live, tab_sim, tab_hist, tab_res, tab_how = st.tabs(
    ["🛰️ Live now", "🎛️ What-if simulator", "📚 Historical events", "📊 Results", "ℹ️ How it works"]
)

# ---------------- live
with tab_live:
    from spacewx_rag.live import build_live_card, get_live_data, latest_snapshot

    top = st.columns([3, 1.2, 1.2])
    with top[0]:
        st.markdown(
            '<div class="ev-title">Current space weather · last 24 hours</div>'
            '<div class="muted">Real-time NOAA SWPC feeds: L1 solar wind, planetary Kp, Kyoto quick-look Dst, F10.7. '
            "A description of recent conditions, not a forecast.</div>",
            unsafe_allow_html=True,
        )
    fetch = top[1].button("🛰️ Fetch latest NOAA data", type="primary", width="stretch")
    gen_live = top[2].button(
        "✨ Generate AI analysis",
        disabled=not up,
        width="stretch",
        help="Runs the local LLM on the live event card (about a minute)",
    )
    if fetch or "live_snap" not in st.session_state:
        try:
            if fetch:
                with st.spinner("Contacting NOAA SWPC…"):
                    snap, status = get_live_data()
            else:
                snap, status = latest_snapshot(), "snapshot"
            card, table = build_live_card(snap)
            st.session_state.update(
                live_card=card,
                live_table=table,
                live_status=status,
                live_time=snap["fetched_at"],
                live_snap=snap,
            )
            st.session_state.pop("gen_live", None)
        except Exception as e:  # noqa: BLE001 - shown to the user, the rest of the app keeps working
            st.error(f"Could not load live data: {e}")
    if "live_card" in st.session_state:
        card, table, status = (
            st.session_state["live_card"],
            st.session_state["live_table"],
            st.session_state["live_status"],
        )
        if status == "fresh":
            st.markdown(
                f'<div class="status ok">🟢 Live data fetched from NOAA SWPC at {st.session_state["live_time"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            note = (
                "showing the last saved snapshot; press Fetch for fresh data"
                if status == "snapshot"
                else status
            )
            st.markdown(
                f'<div class="status warn">🟠 {E(note)} · data from {st.session_state["live_time"]}</div>',
                unsafe_allow_html=True,
            )
        cond = {"quiet": "quiet", "active": "active", "storm": f"{card['class']} storm"}[
            card["type"]
        ]
        st.markdown(
            f'<div style="margin:6px 0 2px 0">{badge(card["class"])} <span class="ev-title" style="margin-left:8px">'
            f'Conditions: {E(cond)}</span></div><div class="muted">Window {card["window"]["start"]} → '
            f"{card['window']['end']} · solar wind monitor {E(card['data_quality']['solar_wind_monitor'])}</div>",
            unsafe_allow_html=True,
        )
        show_kpis(card)
        days = load_params()["live"]["context_days"]
        d = table.loc[table.index.max() - pd.Timedelta(days=days) :]
        st.markdown("#### 🌬️ Solar wind at L1 · last 24 hours, 1-minute data")
        st.plotly_chart(live_solar_wind_plot(st.session_state["live_snap"]), width="stretch")
        st.caption(
            "Measured at the L1 point, about 1.5 million km upstream: it reaches Earth roughly 25 to 100 "
            "minutes later (the event card uses Earth-arrival hourly means)."
        )
        st.markdown(f"#### 🧲 Geomagnetic response · last {days} days")
        st.plotly_chart(
            geomag_plot(d, card["window"]["start"][:-1], card["window"]["end"][:-1]),
            width="stretch",
        )
        st.caption("Shaded band: the 24-hour window the AI describes.")
        show_card_details(card)
        if gen_live:
            generate_flow(card, "gen_live", "what is space weather doing right now?")
        if st.session_state.get("gen_live"):
            st.markdown("### 🤖 AI analysis of current conditions")
            show_generated("gen_live")

# ---------------- what-if simulator
with tab_sim:
    from spacewx_rag.scenario import KP_STEPS, PRESETS, build_scenario_card, kp_label

    SIM_KEYS = {
        "kp": "sim_kp",
        "dst": "sim_dst",
        "speed": "sim_speed",
        "bz": "sim_bz",
        "hours_south": "sim_hours",
        "density": "sim_n",
        "f107": "sim_f107",
    }

    def apply_preset(name: str) -> None:
        for k, v in PRESETS[name].items():
            st.session_state[SIM_KEYS[k]] = v

    if "sim_kp" not in st.session_state:
        apply_preset("Moderate storm")

    st.markdown(
        '<div class="ev-title">What if…? Build your own space weather</div>'
        '<div class="muted">Move the sliders and the event card updates instantly with the same rules as real '
        "events. Then ask the AI what such conditions would mean. This is a hypothetical scenario, not a forecast."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    pc = st.columns(len(PRESETS))
    for col, name in zip(pc, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), width="stretch")

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown(
            '<div class="card"><h4>🧲 Geomagnetic response</h4></div>', unsafe_allow_html=True
        )
        kp = st.select_slider(
            "Kp maximum (planetary index, 0–9)",
            options=KP_STEPS,
            key="sim_kp",
            format_func=lambda k: f"{k:g}  ({kp_label(k)})",
        )
        dst = st.slider(
            "Dst minimum (nT): more negative means a stronger ring current",
            -1000,
            50,
            step=1,
            key="sim_dst",
        )
        st.markdown('<div class="card"><h4>🌬️ Solar wind driver</h4></div>', unsafe_allow_html=True)
        speed = st.slider("Max solar wind speed (km/s)", 250, 1500, step=10, key="sim_speed")
        bz = st.slider(
            "Lowest Bz (nT): negative means southward, the key storm driver",
            -60.0,
            20.0,
            step=0.5,
            key="sim_bz",
        )
        with st.expander("More inputs"):
            hours = st.slider("Hours of southward Bz (≤ −5 nT)", 0, 48, key="sim_hours")
            dens = st.slider("Max proton density (/cm³)", 0.5, 60.0, step=0.5, key="sim_n")
            f107 = st.slider("Solar activity F10.7 (sfu)", 65, 300, key="sim_f107")
    scen = build_scenario_card(kp, dst, speed, bz, hours, dens, f107)
    sig = (kp, dst, speed, bz, hours, dens, f107)
    with right:
        kind = {
            "quiet": "quiet conditions",
            "active": "active, below storm level",
            "storm": f"{scen['class']} geomagnetic storm",
        }[scen["type"]]
        g = scen["geomagnetic"]["noaa_g_level_approx"]
        st.markdown(
            f'<div style="margin:4px 0 6px 0">{badge(scen["class"])} <span class="ev-title" style="margin-left:8px">'
            f'{E(kind)}</span></div><div class="muted">NOAA storm level ≈ <b>{g or "below G1"}</b> · '
            f"electric field vBs {scen['drivers']['vbs_max_mV_per_m']} mV/m · dynamic pressure "
            f"{scen['drivers']['dynamic_pressure_max_nPa_protons_only']} nPa</div>",
            unsafe_allow_html=True,
        )
        show_kpis(scen)
        odd = [c for c in scen["data_quality"]["caveats"] if "unusual" in c]
        for o in odd:
            st.markdown(f'<div class="status warn">🤔 {E(o)}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><h4>Rule-based reading (instant, no AI)</h4>'
            + "".join(f"<div>• {E(h)}</div>" for h in scen["hints"])
            + "</div>",
            unsafe_allow_html=True,
        )
        ask = st.button(
            "✨ Ask the AI what this would mean",
            type="primary",
            disabled=not up,
            width="stretch",
            help="Runs the local LLM on this scenario (about a minute)",
        )
    if ask:
        generate_flow(
            scen,
            "gen_sim",
            f"what would Kp {kp:g}, Dst {dst} nT, wind {speed} km/s and Bz {bz:g} nT mean?",
        )
        st.session_state["gen_sim_sig"] = sig
    if st.session_state.get("gen_sim"):
        st.markdown("### 🤖 What the AI says about this scenario")
        if st.session_state.get("gen_sim_sig") != sig:
            st.markdown(
                '<div class="status warn">🟠 The sliders have changed since this answer was written. '
                "Press the button again to update it.</div>",
                unsafe_allow_html=True,
            )
        show_generated("gen_sim")
    with st.expander("🧾 Scenario event card (everything the model receives)"):
        st.markdown(card_table_html(scen), unsafe_allow_html=True)

# ---------------- historical
with tab_hist:
    card = cs[eid]
    kind = "interval" if card["type"] == "quiet" else "geomagnetic storm"
    st.markdown(
        f'<div>{badge(card["class"])} <span class="ev-title" style="margin-left:8px">{eid} · {card["class"]} {kind}'
        f'</span></div><div class="muted">{card["window"]["start"]} → {card["window"]["end"]} '
        f"({card['window']['hours']} h, UTC) · solar wind shifted to Earth arrival time</div>",
        unsafe_allow_html=True,
    )
    show_kpis(card)
    sub_data, sub_ai, sub_src = st.tabs(["📈 Data", "🤖 AI description & check", "📚 Sources"])
    with sub_data:
        a = aligned()
        t0 = pd.Timestamp(card["window"]["start"][:-1]) - pd.Timedelta(hours=12)
        t1 = pd.Timestamp(card["window"]["end"][:-1]) + pd.Timedelta(hours=12)
        st.plotly_chart(
            storm_plot(a.loc[t0:t1], card["window"]["start"][:-1], card["window"]["end"][:-1]),
            width="stretch",
        )
        show_card_details(card)
    key = f"gen_{eid}"
    with sub_ai:
        head = st.columns([3, 1.2])
        head[0].markdown(
            f"**{RUN_LABELS[run]}** · saved run"
            if key not in st.session_state
            else "**Live re-generation**"
        )
        regen = head[1].button(
            "⚡ Generate live now",
            disabled=not up,
            width="stretch",
            key=f"btn_{eid}",
            help="Runs the local LLM on this event now (about a minute)",
        )
        if regen:
            generate_flow(card, key, f"explain event {eid} ({card['window']['start'][:10]}) to me")
        if st.session_state.get(key):
            if st.button("↩ Back to the saved run", key=f"back_{eid}"):
                del st.session_state[key]
                st.rerun()
            show_generated(key)
        else:
            r = results(run).get((eid, 0))
            if r is None:
                st.warning("No saved output for this event and run.")
            elif not r["valid"]:
                st.error(
                    f"Recorded failure: the model's output was invalid JSON ({r['error']}). It got stuck "
                    "repeating one evidence path; a list-length cap now prevents that."
                )
                with st.expander("Raw model output (end)"):
                    st.code(r["raw"][-1200:])
            else:
                if r.get("prompt_tokens"):
                    st.caption(
                        f"{r['model']} · {r['seconds']:.0f} s · {r['prompt_tokens']} prompt tokens · "
                        f"{r['output_tokens']} output tokens"
                    )
                show_description(r["output"])
                st.markdown("#### ✅ Claim-by-claim check")
                cl = claims()
                rows = cl[
                    (cl["run"] == run) & (cl["event_id"] == eid) & (cl["repeat"] == 0)
                ].to_dict("records")
                show_claims(rows)
    with sub_src:
        s = st.session_state.get(key)
        if s and "passages" in s:
            out, passages = s["output"], s["passages"]
        else:
            r = results(run).get((eid, 0)) or {}
            out, passages = r.get("output") or {}, r.get("passages") or []
        cited = {
            x
            for k in ("possible_physical_interpretation", "potential_impacts")
            for i in out.get(k, [])
            for x in i["sources"]
        }
        show_sources(passages, cited)

# ---------------- results
with tab_res:
    m = metrics()
    first = m[(m["repeat"] == 0) & m["run"].isin(list(RUN_LABELS))]
    summ = (
        first.groupby("run")
        .agg(
            valid=("valid", "mean"),
            groundedness=("groundedness", "mean"),
            completeness=("completeness", "mean"),
            number_errors=("number_errors", "sum"),
        )
        .reindex(list(RUN_LABELS))
    )
    hit3 = pd.read_csv(P.results / "retrieval_summary.csv", index_col=0).loc["all", "hit@3"]
    rag = summ.loc["ollama_rag"]
    st.markdown(
        '<div class="ev-title">Results on 15 evaluation events</div>'
        '<div class="muted">1 super, 5 intense, 4 moderate and 5 quiet events from 2016 to 2025</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="kpis">'
        + kpi(
            "Groundedness · main metric",
            f"{rag['groundedness']:.1%}",
            "92.0% after a manual audit",
            "#2ec4b6",
        )
        + kpi(
            "Same model, no retrieval",
            f"{summ.loc['ollama_no_retrieval', 'groundedness']:.1%}",
            "retrieval adds about 11 points",
            "#ff9f1c",
        )
        + kpi(
            "Completeness",
            f"{rag['completeness']:.1%}",
            "of the hand-written reference points",
            "#4cc9f0",
        )
        + kpi(
            "Number errors",
            f"{int(rag['number_errors'])}",
            "every number matches the data",
            "#c77dff",
        )
        + kpi("Retrieval hit@3", f"{hit3:.0%}", "42 labelled questions", "#ffd166")
        + "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.3, 1])
    with c1:
        by = first.pivot_table(index="class", columns="run", values="groundedness").reindex(
            ["super", "intense", "moderate", "quiet"]
        )[list(RUN_LABELS)]
        colors = {"ollama_rag": "#ff9f1c", "ollama_no_retrieval": "#5a6390", "baseline": "#2ec4b6"}
        fig = go.Figure(
            [
                go.Bar(name=RUN_LABELS[c], x=by.index, y=by[c], marker_color=colors[c])
                for c in by.columns
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            barmode="group",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.02)",
            yaxis={"range": [0, 1.05], "title": "groundedness"},
            title="Groundedness by event class",
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            legend={"orientation": "h", "y": -0.15},
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        table = summ.rename(index=RUN_LABELS).rename(
            columns={"valid": "valid JSON", "number_errors": "number errors"}
        )
        st.dataframe(
            table.style.format(
                {
                    "valid JSON": "{:.0%}",
                    "groundedness": "{:.1%}",
                    "completeness": "{:.1%}",
                    "number errors": "{:.0f}",
                }
            ),
            width="stretch",
        )
        st.markdown(
            '<div class="card"><h4>What it shows</h4>'
            "<div>📚 <b>Retrieval helps:</b> 0.83 → 0.94 groundedness, 0.75 → 0.86 completeness</div>"
            "<div>🔢 <b>Numbers are always right</b>, and quiet days stay quiet</div>"
            "<div>⚠️ <b>Failures are about causes and impacts</b>, and the checker catches them</div>"
            "<div>🧪 <b>The baseline scores 100% by construction</b>: groundedness measures support, not usefulness</div>"
            "<div>🧑‍⚖️ <b>The judge is a small model</b>, so it was audited by hand (19 of 21 agree)</div></div>",
            unsafe_allow_html=True,
        )
    cons = P.results / "consistency.csv"
    if cons.exists():
        with st.expander("🔁 Consistency: 5 events × 3 runs at temperature 0.7"):
            st.dataframe(
                pd.read_csv(cons).drop(columns=["run"]).round(2), hide_index=True, width="stretch"
            )

# ---------------- how it works
with tab_how:
    steps = [
        (
            "1",
            "🛰️ Data",
            (
                "DSCOVR solar wind and magnetic field, Kyoto Dst, GFZ Kp, CelesTrak F10.7, with NASA OMNI for "
                "gaps. 87,024 hours (2016–2026) on one hourly timeline, shifted to Earth arrival time."
            ),
        ),
        (
            "2",
            "🌩️ Events",
            (
                "202 storms found from Dst plus 100 quiet control windows, each with drivers, indices and "
                "data-quality features."
            ),
        ),
        (
            "3",
            "🧮 Event card",
            "The facts, rule-based hints and caveats for one event. Never a cause or an impact.",
        ),
        (
            "4",
            "📚 Retrieval",
            (
                "21 public references (NOAA, GFZ, Kyoto, NASA, a 2025 paper) → 290 chunks → local "
                "embeddings → Chroma. Several targeted queries per event."
            ),
        ),
        (
            "5",
            "🤖 Generation",
            (
                "qwen2.5:7b-instruct runs locally through Ollama: free and offline. The output schema "
                "only allows real card fields and retrieved passages as citations."
            ),
        ),
        (
            "6",
            "✅ Checking",
            (
                "Numbers against the card, causes and impacts against the cited passages (an LLM judge "
                "with a verified quote, plus a NOAA G-level check), no forecasts, quiet days kept quiet."
            ),
        ),
    ]
    st.markdown(
        '<div class="steps">'
        + "".join(
            f'<div class="step"><div class="n">{n}</div><b>{t}</b><span class="muted">{E(d)}</span></div>'
            for n, t, d in steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        '<div class="card"><h4>Scope</h4>Explains past and current conditions; it does not forecast. '
        "Live mode uses NOAA's real-time feeds (unvalidated) and is a demonstration, not part of the evaluation."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Data: NOAA NCEI and SWPC, GFZ Potsdam (CC BY 4.0), WDC for Geomagnetism Kyoto (doi:10.17593/14515-74000), "
        "CelesTrak, NASA OMNIWeb. Document licences: knowledge_base/raw_docs/MANIFEST.csv. "
        "Code: src/spacewx_rag/ · python -m spacewx_rag <stage> · 58 tests."
    )
