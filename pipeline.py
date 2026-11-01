"""Orchestrates: sanitize -> PII mask -> budget -> injection -> crew -> groundedness -> review -> final."""
import config  # noqa: F401
import re
from typing import Optional
from pydantic import BaseModel, Field
from config import MAX_QUERY_CHARS, TOKEN_BUDGET
from crew import run_crew, CrewResponse
from review import review_draft, Verdict
from rag import FALLBACK, ensure_index
from security import (sanitize, mask_pii, detect_injection, groundedness_check, check_budget, BudgetExceeded)
from mock_llm import reset_usage, USAGE


class InputTooLarge(Exception): ...


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    refused: bool = False
    guardrails_fired: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    order_id: Optional[str] = None
    order_status: Optional[str] = None
    escalation_score: Optional[float] = None
    escalate: Optional[bool] = None
    cache_hit: bool = False
    review: Optional[Verdict] = None
    tokens_used: int = 0
    masked_query: str = ""


def _refuse(masked, fired, msg, tokens=0):
    return AskResponse(answer=msg, grounded=False, refused=True, guardrails_fired=fired, masked_query=masked, tokens_used=tokens)


def handle(query: str) -> AskResponse:
    if len(str(query)) > MAX_QUERY_CHARS:
        raise InputTooLarge(f"query longer than {MAX_QUERY_CHARS} chars")
    ensure_index()
    masked, pii = mask_pii(query)
    fired = [f"pii_masking:{'+'.join(pii)}"] if pii else []
    if not masked:
        return _refuse(masked, fired + ["empty_input"], "Please type your question.")
    check_budget(masked)                                   # Runtime layer: raises BudgetExceeded BEFORE any work
    inj, _ = detect_injection(masked)
    if inj:
        return _refuse(masked, fired + ["prompt_injection"], "I can't help with that request. Please ask a Nykaa support question.")
    reset_usage(cap=int(TOKEN_BUDGET * 1.3))               # hard ceiling during the run (defence in depth)
    try:
        r: CrewResponse = run_crew(masked)
        used = USAGE["tokens"]
    finally:
        reset_usage(cap=None)                              # never leave a stale cap behind
    if not r.grounded:
        return _refuse(masked, fired + ["groundedness"], FALLBACK, used)
    ok, ratio = groundedness_check(r.answer, r.context)
    if not ok:
        return _refuse(masked, fired + [f"groundedness:{ratio}"], FALLBACK, used)
    v = review_draft(r.answer, r.context)                  # Task 14
    final, _ = mask_pii(v.final_answer)                    # output-side PII safety net
    return AskResponse(answer=final, grounded=True, guardrails_fired=fired, sources=r.sources, context=r.context, order_id=r.order_id,
                       order_status=r.order_status, escalation_score=r.escalation_score, escalate=r.escalate,
                       cache_hit=r.cache_hit, review=v, tokens_used=used, masked_query=masked)
