"""Prompt assembly, generators and the structured output (notebook 06)."""

from spacewx_rag.generation.generators import (
    BaselineGenerator,
    GenerationResult,
    OllamaGenerator,
    get_generator,
)
from spacewx_rag.generation.pipeline import generate_events, load_results, result_path
from spacewx_rag.generation.prompts import judge_prompt, system_prompt, user_prompt
from spacewx_rag.generation.schema import EventDescription, json_schema

__all__ = [
    "BaselineGenerator",
    "EventDescription",
    "GenerationResult",
    "OllamaGenerator",
    "generate_events",
    "get_generator",
    "json_schema",
    "judge_prompt",
    "load_results",
    "result_path",
    "system_prompt",
    "user_prompt",
]
