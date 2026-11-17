"""Prompt assembly from the templates in prompts/."""

from __future__ import annotations

import json
from string import Template

from spacewx_rag.utils.config import Paths

NO_PASSAGES = (
    "No passages were provided. Every cause, impact or comparison has no supporting source."
)


def load_template(name: str) -> Template:
    return Template((Paths().prompts / name).read_text(encoding="utf-8"))


def system_prompt() -> str:
    return load_template("system.md").template.strip()


def model_card(card: dict) -> dict:
    """The card as the model sees it: never with `evaluation_role` (decision 32)."""
    return {k: v for k, v in card.items() if k != "evaluation_role"}


def format_passages(passages: list[dict]) -> str:
    if not passages:
        return NO_PASSAGES
    blocks = []
    for p in passages:
        head = p["title"] + (f" - {p['section']}" if p.get("section") else "")
        blocks.append(f"[{p['chunk_id']}] ({head})\n{p['text'].strip()}")
    return "\n\n".join(blocks)


def user_prompt(card: dict, passages: list[dict]) -> str:
    c = model_card(card)
    return (
        load_template("event_user.md")
        .substitute(
            card=json.dumps(c, indent=1, ensure_ascii=False),
            passages=format_passages(passages),
            event_id=c["event_id"],
        )
        .strip()
    )


def judge_prompt(claim: str, passages: list[dict]) -> str:
    return (
        load_template("judge_support.md")
        .substitute(claim=claim, passages=format_passages(passages))
        .strip()
    )
