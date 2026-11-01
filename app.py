"""Tasks 11-12 - FastAPI: POST /ask, POST /add-document, GET /health, WS /ws/chat + JSON-Lines logging."""
import config  # noqa: F401
import hmac, json, re, threading, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from config import LOG_DIR, API_KEY, MAX_QUERY_CHARS
import rag
from memory import ask_in_session, new_session_id
from pipeline import handle, AskResponse, InputTooLarge
from security import BudgetExceeded, mask_pii, detect_injection, rate_limited

LOG_FILE = LOG_DIR / "requests.jsonl"
_LW = threading.Lock()


def write_log(entry: dict):
    """One JSON line per request. Every string is passed through the SAME PII masker the model sees."""
    safe = {k: (mask_pii(v)[0] if isinstance(v, str) else v) for k, v in entry.items()}
    with _LW, open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app):
    rag.ensure_index(); rag.get_threshold()   # warm-up: index + calibration
    yield


app = FastAPI(title="Nykaa Support Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class AddDocRequest(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    text: str = Field(min_length=20, max_length=2000)
    doc_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$")


class AddDocResponse(BaseModel):
    doc_id: str
    chunks_indexed: dict


@app.middleware("http")
async def log_mw(request: Request, call_next):
    trace, t0, status = uuid.uuid4().hex, time.perf_counter(), 500
    request.state.trace_id, request.state.log = trace, {}
    try:
        resp = await call_next(request); status = resp.status_code
        resp.headers["X-Trace-Id"] = trace
        return resp
    finally:
        write_log({"ts": datetime.now(timezone.utc).isoformat(), "trace_id": trace, "transport": "http",
                   "method": request.method, "path": request.url.path, "status": status,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 2), **request.state.log})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):   # never leak internals
    request.state.log["error"] = type(exc).__name__
    return JSONResponse({"detail": "internal error", "trace_id": request.state.trace_id}, status_code=500)


def _guard(request: Request):
    if rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limit exceeded")


def _run(fn, *a):
    try: return fn(*a)
    except BudgetExceeded as e: raise HTTPException(413, f"token budget exceeded: {e}")
    except InputTooLarge as e: raise HTTPException(413, str(e))


@app.post("/ask", response_model=AskResponse, response_model_exclude={"context"})
def ask(req: AskRequest, request: Request):
    _guard(request)
    res = _run(ask_in_session, req.query, req.session_id) if req.session_id else _run(handle, req.query)
    request.state.log.update(query=res.masked_query, refused=res.refused, guardrails=res.guardrails_fired,
                             cache_hit=res.cache_hit, tokens_used=res.tokens_used, session=bool(req.session_id))
    return res


@app.post("/add-document", response_model=AddDocResponse)
def add_document(req: AddDocRequest, request: Request, x_api_key: Optional[str] = Header(default=None)):
    _guard(request)
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401, "invalid API key")
    n = len(rag.split_sentences(req.text))
    if not 2 <= n <= 5: raise HTTPException(422, "document must have 2-5 sentences")
    if mask_pii(req.title + " " + req.text)[1]: raise HTTPException(422, "document contains PII; not stored")
    if detect_injection(req.title + " " + req.text)[0]: raise HTTPException(400, "document rejected: prompt-injection content")
    doc_id = req.doc_id or "USR-" + uuid.uuid4().hex[:8]
    if re.match(r"(?i)^KB\d+$", doc_id): raise HTTPException(409, "reserved id: built-in knowledge-base documents cannot be overwritten")
    counts = rag.add_documents([{"id": doc_id, "title": req.title, "text": req.text}])
    request.state.log.update(doc_id=doc_id, title=req.title, chars=len(req.text))
    return AddDocResponse(doc_id=doc_id, chunks_indexed=counts)


@app.get("/health")
def health():
    return {"status": "ok", "embedder": __import__("embedder").get_embedder().backend, "strategy": rag.deployed_strategy(), "stats": rag.STATS}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    sid, client = new_session_id(), (ws.client.host if ws.client else "ws")
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": raise WebSocketDisconnect(msg.get("code", 1000))
            text, trace, t0 = msg.get("text"), uuid.uuid4().hex, time.perf_counter()
            entry = {"ts": datetime.now(timezone.utc).isoformat(), "trace_id": trace, "transport": "ws", "path": "/ws/chat", "session": sid}
            if text is None: out, entry["status"] = {"error": "text frames only"}, 400
            elif rate_limited(client): out, entry["status"] = {"error": "rate limit exceeded"}, 429
            else:
                try:
                    res = await run_in_threadpool(ask_in_session, text, sid)
                    out, entry["status"] = res.model_dump(exclude={"context"}), 200
                    entry.update(query=res.masked_query, refused=res.refused, guardrails=res.guardrails_fired, cache_hit=res.cache_hit)
                except (BudgetExceeded, InputTooLarge) as e: out, entry["status"] = {"error": str(e)}, 413
                except Exception as e:                       # one bad message must not kill the socket / server
                    out, entry["status"] = {"error": "internal error"}, 500; entry["error"] = type(e).__name__
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            write_log(entry)
            await ws.send_json({"trace_id": trace, **out})
    except (WebSocketDisconnect, RuntimeError):              # client left mid-conversation: server keeps serving others
        write_log({"ts": datetime.now(timezone.utc).isoformat(), "transport": "ws", "path": "/ws/chat",
                   "trace_id": uuid.uuid4().hex, "event": "client_disconnected", "session": sid})
