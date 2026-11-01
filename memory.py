"""Task 8 - session memory via LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory."""
import config  # noqa: F401
import re, uuid, threading
from collections import OrderedDict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from pipeline import handle, AskResponse, InputTooLarge
from security import mask_pii
from config import MAX_QUERY_CHARS

MAX_SESSIONS, MAX_MSGS = 1000, 20
_STORE: "OrderedDict[str, InMemoryChatMessageHistory]" = OrderedDict()
_L = threading.Lock()
_ID = re.compile(r"NYK-\d{4}", re.I)
_REF = re.compile(r"\b(it|this order|that order|my order|the order)\b", re.I)
_RECALL = re.compile(r"what did i (?:ask|say)(?: first| earlier| before)?|first question", re.I)


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    with _L:
        if session_id not in _STORE:
            _STORE[session_id] = InMemoryChatMessageHistory()
            while len(_STORE) > MAX_SESSIONS: _STORE.popitem(last=False)   # bound memory (DoS guard)
        _STORE.move_to_end(session_id)
        h = _STORE[session_id]
        if len(h.messages) > MAX_MSGS: h.messages = h.messages[-MAX_MSGS:]
        return h


def _turn(inp: dict) -> dict:
    q, hist = inp["input"], inp["history"]
    humans = [m.content for m in hist if m.type == "human"]
    if _RECALL.search(q):
        ans = f"Your first question was: \"{humans[0]}\"" if humans else "This is the start of our conversation - I have no earlier question."
        return {"output": ans, "response": AskResponse(answer=ans, grounded=True, guardrails_fired=["memory_recall"])}
    if not _ID.search(q) and _REF.search(q):
        prev = [i for m in reversed(hist) for i in _ID.findall(m.content)]
        if prev: q = f"{q} (order {prev[0].upper()})"
        elif re.search(r"\b(this|that|my|the) order\b", q, re.I):
            ans = "I don't have an order in this conversation yet. Please share your order id (e.g. NYK-1005)."
            return {"output": ans, "response": AskResponse(answer=ans, grounded=False, guardrails_fired=["needs_order_id"])}
    r = handle(q)
    return {"output": r.answer, "response": r}


chat = RunnableWithMessageHistory(RunnableLambda(_turn), get_history, input_messages_key="input",
                                  history_messages_key="history", output_messages_key="output")


def ask_in_session(query: str, session_id: str) -> AskResponse:
    if len(query) > MAX_QUERY_CHARS: raise InputTooLarge(f"query longer than {MAX_QUERY_CHARS} chars")
    masked, pii = mask_pii(query)   # memory stores the MASKED text only - raw PII never enters history
    out = chat.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
    r = out["response"]
    if pii and not any(g.startswith("pii_masking") for g in r.guardrails_fired):
        r.guardrails_fired.insert(0, f"pii_masking:{'+'.join(pii)}")
    return r


def new_session_id() -> str: return uuid.uuid4().hex
