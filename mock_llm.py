"""MOCK_LLM for CrewAI: deterministic, keyless, networkless. Extends crewai BaseLLM (documented extension point).
Avoids both known pitfalls:
 (1) Tool results are read ONLY from the model's own previous assistant message - never the system-prompt
     template (which literally contains 'Observation: the result of the action').
 (2) Tool arguments are chosen from the tool's declared argument schema ("required": [...]), not tool-name matching."""
import config  # noqa: F401
import json, re, threading
from crewai.llms.base_llm import BaseLLM
from rag import FALLBACK
from security import BudgetExceeded, estimate_tokens

USAGE = {"tokens": 0, "calls": 0, "cap": None}
_ULOCK = threading.Lock()
_ORDER_ID = re.compile(r"NYK-\d{4}", re.I)


def reset_usage(cap=None):
    with _ULOCK: USAGE.update(tokens=0, calls=0, cap=cap)


def _account(text_in, text_out):
    with _ULOCK:
        USAGE["tokens"] += estimate_tokens(text_in) + estimate_tokens(text_out); USAGE["calls"] += 1
        if USAGE["cap"] is not None and USAGE["tokens"] > USAGE["cap"]:
            raise BudgetExceeded(f"runtime usage {USAGE['tokens']} > cap {USAGE['cap']}")


def _final(x): return f"Thought: I now know the final answer\nFinal Answer: {x}"


class MockLLM(BaseLLM):
    def __init__(self, **kw): super().__init__(model="mock-llm", **kw)
    def supports_function_calling(self) -> bool: return False   # force text ReAct path
    def supports_stop_words(self) -> bool: return False
    def get_context_window_size(self) -> int: return 8192

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        sys_txt = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        convo = [m for m in msgs if m["role"] != "system"]
        user_txt = "\n".join(m["content"] for m in convo if m["role"] == "user")
        out = self._decide(sys_txt, convo, user_txt)
        _account(sys_txt + user_txt, out)
        return out

    def _decide(self, sys_txt, convo, user_txt):
        # (1) tool already ran? read Observation from the MODEL'S OWN assistant message only.
        asst = [m["content"] for m in convo if m["role"] == "assistant"]
        if asst and "Observation:" in asst[-1]:
            return _final(asst[-1].rsplit("Observation:", 1)[1].strip())
        qm = re.search(r"USER_QUERY: (.*)", user_txt)
        query = qm.group(1).strip() if qm else ""
        # (2) agent has a tool: choose args from the declared schema
        tm = re.search(r"Tool Name: (\w+).*?\"required\": \[\s*\"(\w+)\"", sys_txt, re.S)
        if tm:
            tool, arg = tm.groups()
            if arg == "record_id":
                oid = _ORDER_ID.search(query)
                if not oid: return _final('ORDER_JSON={"skipped": true}')
                val = oid.group(0).upper()
            else:
                val = query
            return f"Thought: I should call the tool.\nAction: {tool}\nAction Input: {json.dumps({arg: val})}"
        # (3) no tools -> Composer: merge upstream outputs
        return _final(self._compose(user_txt))

    @staticmethod
    def _compose(ctx):
        def grab(tag):
            m = re.search(rf"{tag}=(\{{.*\}})", ctx)
            try: return json.loads(m.group(1)) if m else {}
            except json.JSONDecodeError: return {}
        rag, order = grab("RAG_JSON"), grab("ORDER_JSON")
        r = dict(answer=FALLBACK, grounded=False, sources=[], context=[], order_id=None, order_status=None,
                 escalation_score=None, escalate=None, refused=False, guardrail=None, cache_hit=bool(rag.get("cache_hit")))
        parts, context = [], []
        if order.get("status"):
            line = (f"Order {order['record_id']} is {order['status']} with value {order['order_value_inr']} INR. "
                    f"Escalation score is {order['escalation_score']} against threshold {order['threshold']}, so "
                    + ("escalation is recommended." if order["escalate"] else "escalation is not needed."))
            parts.append(line); context.append(line)
            r.update(order_id=order["record_id"], order_status=order["status"],
                     escalation_score=order["escalation_score"], escalate=order["escalate"], grounded=True)
        elif order.get("error"):
            parts.append("I could not find a valid order for that order id.")
        # with an order record, add policy text only on a strong match (avoids irrelevant filler)
        if rag.get("grounded") and (not order.get("status") or rag.get("top_score", 0) >= 1.5 * rag.get("threshold", 1)):
            parts.append(rag["answer"]); context += rag["context"]
            r.update(sources=rag["sources"], grounded=True)
        if parts: r["answer"] = " ".join(parts)
        r["context"] = context
        return json.dumps(r)
