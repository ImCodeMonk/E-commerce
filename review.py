"""Task 14 - Autogen RoundRobinGroupChat review stage (Policy-Compliance-Reviewer + Final-Editor), keyless."""
import config  # noqa: F401
import asyncio, re, threading
from typing import Sequence
from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from rag import split_sentences, FALLBACK
from security import support_ratio, estimate_tokens

MIN_SUPPORT = 0.7


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


def _text(m):
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


class MockReviewClient(ChatCompletionClient):
    """Deterministic stand-in for an LLM. Reviewer -> text verdict token; Editor -> JSON Verdict."""
    def __init__(self): self._u = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self): return ModelInfo(vision=False, function_calling=False, json_output=True,
                                           family="unknown", structured_output=True)
    @property
    def capabilities(self): return self.model_info
    def count_tokens(self, messages, *, tools=()): return sum(estimate_tokens(_text(m)) for m in messages)
    def remaining_tokens(self, messages, *, tools=()): return 8000
    def actual_usage(self): return self._u
    def total_usage(self): return self._u
    async def close(self): return None

    async def create_stream(self, messages, **kw):
        yield await self.create(messages, **kw)

    async def create(self, messages: Sequence, *, tools=(), tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None) -> CreateResult:
        convo = "\n".join(_text(m) for m in messages if type(m).__name__ != "SystemMessage")
        m = re.search(r"DRAFT:\n(.*?)\nCONTEXT:\n(.*?)\n<<<END_TASK>>>", convo, re.S)
        draft, ctx = (m.group(1), m.group(2)) if m else ("", "")
        sents = split_sentences(draft)
        good = [s for s in sents if support_ratio(s, ctx) >= MIN_SUPPORT]
        bad = [s for s in sents if s not in good]
        if json_output is None:   # ---- Policy-Compliance-Reviewer
            out = ("[[REVIEW:APPROVE]] Every sentence is supported by the retrieved context." if not bad else
                   "[[REVIEW:REVISE]] Unsupported claim(s): " + " | ".join(bad))
        else:                     # ---- Final-Editor (structured output)
            tags = re.findall(r"\[\[REVIEW:(APPROVE|REVISE)\]\]", convo)
            if tags and tags[-1] == "APPROVE" and not bad:
                v = Verdict(approved=True, final_answer=draft, reason="Draft fully grounded in context; approved unchanged.")
            else:
                v = Verdict(approved=False, final_answer=" ".join(good) if good else FALLBACK,
                            reason=f"Removed {len(bad)} ungrounded sentence(s) not supported by the retrieved context.")
            out = v.model_dump_json()
        pt = self.count_tokens(messages); ct = estimate_tokens(out)
        self._u = RequestUsage(prompt_tokens=self._u.prompt_tokens + pt, completion_tokens=self._u.completion_tokens + ct)
        return CreateResult(finish_reason="stop", content=out, usage=RequestUsage(prompt_tokens=pt, completion_tokens=ct), cached=False)


def build_team():
    client = MockReviewClient()
    reviewer = AssistantAgent("Policy_Compliance_Reviewer", model_client=client,
                              system_message="Check every sentence of the DRAFT is supported by CONTEXT. Reply APPROVE or REVISE.")
    editor = AssistantAgent("Final_Editor", model_client=client, output_content_type=Verdict,
                            system_message="Use the review to approve the draft unchanged or revise it. Output a Verdict.")
    return RoundRobinGroupChat([reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])


async def review_async(draft: str, context: list[str]) -> Verdict:
    task = "DRAFT:\n" + draft + "\nCONTEXT:\n" + "\n".join(context) + "\n<<<END_TASK>>>"
    res = await build_team().run(task=task)
    c = res.messages[-1].content
    return c if isinstance(c, Verdict) else Verdict.model_validate_json(c)


def review_draft(draft: str, context: list[str]) -> Verdict:
    """Sync wrapper; runs the event loop in its own thread so it is safe under FastAPI/uvicorn."""
    box = {}
    def _run():
        try: box["v"] = asyncio.run(review_async(draft, context))
        except Exception as e: box["e"] = e
    t = threading.Thread(target=_run); t.start(); t.join()
    if "e" in box: raise box["e"]
    return box["v"]
