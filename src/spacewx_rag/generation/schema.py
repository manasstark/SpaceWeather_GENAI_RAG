"""The structured output every generator must return (docs/architecture.md section 2.5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ImpactArea = Literal[
    "power systems", "satellites and spacecraft", "GNSS navigation", "HF radio", "aurora", "other"
]


class Characteristic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(
        description="One observed fact about the event, using the card's numbers exactly."
    )
    evidence: list[str] = Field(description="Card field paths that support it, e.g. 'dst.min_nT'.")


class Interpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(description="A possible physical explanation drawn from the passages.")
    sources: list[str] = Field(
        description="Chunk ids of the passages that support it, e.g. 'd03_c001'."
    )


class Impact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    area: ImpactArea
    statement: str = Field(
        description="A potential impact drawn from the passages, at this storm level."
    )
    sources: list[str] = Field(description="Chunk ids of the passages that support it.")


class EventDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    summary: str = Field(description="Two to three sentences.")
    characteristics: list[Characteristic]
    possible_physical_interpretation: list[Interpretation]
    potential_impacts: list[Impact]
    caveats: list[str]
    unsupported_or_unknown: list[str] = Field(
        description="Things that cannot be said from the card and passages, e.g. 'cause: no supporting source'."
    )

    def all_text(self) -> str:
        parts = [self.summary]
        parts += [c.statement for c in self.characteristics]
        parts += [i.statement for i in self.possible_physical_interpretation]
        parts += [i.statement for i in self.potential_impacts]
        parts += self.caveats + self.unsupported_or_unknown
        return "\n".join(parts)


def json_schema() -> dict:
    return EventDescription.model_json_schema()
