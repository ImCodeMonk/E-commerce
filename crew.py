"""Tasks 7, 9 - CrewAI crew (Retrieval, Lookup, Composer) + Pydantic structured output."""
import config  # noqa: F401
import json, re, threading
from typing import Optional
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field
from mock_llm import MockLLM
from tools import RagLookupTool, CheckOrderStatusTool, enforce_least_autonomy


class CrewResponse(BaseModel):
    """Task 9: every crew response must conform to this schema."""
    answer: str
    grounded: bool
    sources: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    order_id: Optional[str] = None
    order_status: Optional[str] = None
    escalation_score: Optional[float] = Field(default=None, ge=0, le=1)
    escalate: Optional[bool] = None
    refused: bool = False
    guardrail: Optional[str] = None
    cache_hit: bool = False


def make_agent(role, goal, backstory, tools):
    return Agent(role=role, goal=goal, backstory=backstory, llm=MockLLM(), allow_delegation=False,
                 verbose=False, max_iter=4, tools=enforce_least_autonomy(role, tools))


def build_crew(masked_query):
    retr = make_agent("Retrieval Agent", "Find policy facts in the knowledge base", "Policy search specialist.", [RagLookupTool()])
    look = make_agent("Lookup Agent", "Look up order records", "Only agent allowed to read order data.", [CheckOrderStatusTool()])
    comp = make_agent("Response Composer", "Merge findings into one grounded answer", "Writes the final customer reply.", [])
    t1 = Task(description=f"USER_QUERY: {masked_query}\nSearch the policy knowledge base.", expected_output="RAG_JSON line", agent=retr)
    t2 = Task(description=f"USER_QUERY: {masked_query}\nLook up the order if an order id is present.", expected_output="ORDER_JSON line", agent=look)
    t3 = Task(description="Compose the final answer as JSON matching the CrewResponse schema.", expected_output="CrewResponse JSON",
              agent=comp, context=[t1, t2], output_pydantic=CrewResponse)
    return Crew(agents=[retr, look, comp], tasks=[t1, t2, t3], process=Process.sequential, verbose=False, memory=False)


def audit_crew(crew):
    """Evidence for least autonomy: which agent holds which tools."""
    return {a.role: [t.name for t in a.tools] for a in crew.agents}


_CREW_LOCK = threading.Lock()  # serialise runs: keeps tool log + budget accounting exact


def run_crew(masked_query) -> CrewResponse:
    with _CREW_LOCK:
        result = build_crew(masked_query).kickoff()
    raw = re.sub(r"^```(?:json)?|```$", "", (result.raw or "").strip(), flags=re.M).strip()
    return CrewResponse.model_validate_json(raw)  # explicit validation in code (Task 9)
