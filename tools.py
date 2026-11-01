"""Tasks 6, 15 - tools, escalation score, least-autonomy enforcement."""
import config  # noqa: F401
import json, re, threading
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dataset import ORDERS, ORDER_INDEX
from rag import grounded_answer

# ---- Task 6: designed escalation score ----
W_DELAY, W_RECENCY = 0.6, 0.4


def escalation_score(o):
    """score = 0.6*delayed + 0.4*(days_since_created/30)  in [0,1]."""
    return round(W_DELAY * float(o["delayed_shipment"]) + W_RECENCY * o["days_since_created"] / 30, 3)


def _pct(vals, p):
    s = sorted(vals); return s[int(p * (len(s) - 1))]


SCORES = [escalation_score(o) for o in ORDERS]
# Threshold = 80th percentile of OUR dataset's scores -> only the top ~20% riskiest orders are escalated.
ESCALATION_THRESHOLD = _pct(SCORES, 0.80)
_ID = re.compile(r"NYK-\d{4}")


def check_order_status(record_id: str) -> dict:
    rid = str(record_id).strip().upper()
    if not _ID.fullmatch(rid):  # blocks injection-style / malformed ids
        return {"error": "invalid_record_id"}
    o = ORDER_INDEX.get(rid)
    if o is None:
        return {"error": "not_found", "record_id": rid}
    sc = escalation_score(o)
    return {"record_id": rid, "status": o["status"], "order_value_inr": o["order_value_inr"],
            "escalation_score": sc, "escalate": sc >= ESCALATION_THRESHOLD, "threshold": ESCALATION_THRESHOLD}


# ---- tool-call audit log (evidence that tools really ran) ----
TOOL_LOG, _LOG_LOCK = [], threading.Lock()


def _log(name, args):
    with _LOG_LOCK: TOOL_LOG.append({"tool": name, "args": args})


class RagArgs(BaseModel):
    query: str = Field(..., description="The customer's policy question")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Search the Nykaa policy knowledge base and return a grounded answer with sources."
    args_schema: Type[BaseModel] = RagArgs

    def _run(self, query: str) -> str:
        _log("rag_lookup", {"query": query})
        return "RAG_JSON=" + json.dumps(grounded_answer(query))


class OrderArgs(BaseModel):
    record_id: str = Field(..., description="Order id like NYK-1005")


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = "Look up an order's status, value and escalation score by record_id."
    args_schema: Type[BaseModel] = OrderArgs

    def _run(self, record_id: str) -> str:
        _log("check_order_status", {"record_id": record_id})
        return "ORDER_JSON=" + json.dumps(check_order_status(record_id))


# ---- Task 15: Application-layer least autonomy ----
TOOL_PERMISSIONS = {"check_order_status": {"Lookup Agent"}, "rag_lookup": {"Retrieval Agent"}}


def enforce_least_autonomy(role, tools):
    for t in tools:
        if role not in TOOL_PERMISSIONS.get(t.name, set()):
            raise PermissionError(f"Least-autonomy violation: agent '{role}' may not hold tool '{t.name}'")
    return tools
