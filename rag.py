"""Tasks 3-5 + 16 - chunking, ChromaDB indexing, grounded generation, evaluation, cache."""
import config  # noqa: F401  (must be first)
import json, re, shutil
import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, CALIB_FILE, TOP_K
from embedder import get_embedder, tokens
from kb import KB_DOCS

COLL = {"fixed": "nykaa_fixed", "sentence": "nykaa_sentence"}
FALLBACK = "I don't know based on the Nykaa knowledge base. Please contact support for help with this."
STATS = {"retrieval_calls": 0, "generation_calls": 0, "cache_hits": 0}
_CACHE = {}
_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return _client


# ---------- chunking ----------
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip()) if s.strip()]


def chunk_fixed(text, size=30, overlap=10):
    w = text.split(); step = size - overlap
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w) - overlap, 1), step)]


def chunk_doc(doc, strategy):
    parts = chunk_fixed(doc["text"]) if strategy == "fixed" else split_sentences(doc["text"])
    return [(f"{doc['id']}::{strategy}::{i}", p, {"doc_id": doc["id"], "title": doc["title"], "chunk_index": i, "parent": doc["text"]})
            for i, p in enumerate(parts)]


def _upsert(strategy, docs):
    col = get_client().get_or_create_collection(COLL[strategy], metadata={"hnsw:space": "cosine"})
    rows = [r for d in docs for r in chunk_doc(d, strategy)]
    col.upsert(ids=[r[0] for r in rows], documents=[r[1] for r in rows],
               embeddings=get_embedder().encode([r[1] for r in rows]), metadatas=[r[2] for r in rows])
    return len(rows)


def build_index(docs=None, reset=False):
    global _client
    if reset:
        for n in COLL.values():
            try: get_client().delete_collection(n)
            except Exception: pass
        _CACHE.clear()
    return {s: _upsert(s, docs or KB_DOCS) for s in COLL}


def add_documents(docs):
    n = {s: _upsert(s, docs) for s in COLL}
    _CACHE.clear()  # stale answers must not survive a KB change
    return n


def ensure_index():
    try:
        if all(get_client().get_collection(n).count() > 0 for n in COLL.values()):
            return
    except Exception:
        pass
    build_index()


# ---------- retrieval ----------
def retrieve(query, strategy=None, k=TOP_K):
    strategy = strategy or deployed_strategy()
    col = get_client().get_collection(COLL[strategy])
    res = col.query(query_embeddings=get_embedder().encode([query]), n_results=min(k, col.count()))
    return [{"text": t, "doc_id": m["doc_id"], "title": m["title"], "parent": m["parent"], "score": round(1 - d, 4)}
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


# ---------- calibration (Task 4) ----------
IN_SCOPE = ["What is the return window for Apparel?", "How long does a COD refund take?",
            "Can I cancel an order after it is shipped?", "How many loyalty points can I redeem?",
            "What warranty does Electronics have?"]
OUT_SCOPE = ["Who won the cricket world cup final?", "Give me a recipe for butter chicken",
             "Explain quantum entanglement simply"]


def calibrate():
    ensure_index()
    out = {"measured": {}, "threshold": {}}
    for s in COLL:
        i = [retrieve(q, s, 1)[0]["score"] for q in IN_SCOPE]
        o = [retrieve(q, s, 1)[0]["score"] for q in OUT_SCOPE]
        assert min(i) > max(o), f"clusters overlap for {s}: in={i} out={o}"
        out["measured"][s] = {"in_scope": i, "out_of_scope": o}
        out["threshold"][s] = round((min(i) + max(o)) / 2, 4)
    prev = _load_calib()
    out["strategy"] = prev.get("strategy", "sentence")
    CALIB_FILE.write_text(json.dumps(out, indent=2))
    return out


def _load_calib():
    return json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}


def deployed_strategy():
    return _load_calib().get("strategy", "sentence")


def get_threshold(strategy=None):
    strategy = strategy or deployed_strategy()
    c = _load_calib()
    if "threshold" not in c:
        c = calibrate()
    return c["threshold"][strategy]


# ---------- grounded generation (Task 4) + cache (Task 16) ----------
def normalize_query(q):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", q.lower()).split())


def expand(chunk):
    """Small-to-big: turn a (possibly mid-sentence) chunk into the complete parent sentences it touches."""
    c = " ".join(chunk["text"].lower().split()); out = []
    for sent in split_sentences(chunk["parent"]):
        sl = " ".join(sent.lower().split()); w = sl.split()
        if sl in c or c in sl or " ".join(w[:5]) in c or " ".join(w[-5:]) in c: out.append(sent)
    return out or [chunk["text"]]


def context_sentences(chunks):
    return list(dict.fromkeys(s for c in chunks for s in expand(c)))


def mock_generate(query, sents):
    """MOCK_LLM generator: extractive, uses ONLY the retrieved context sentences, ranked by embedding similarity."""
    if not sents: return FALLBACK
    emb = get_embedder(); q = emb.encode([query])[0]
    sims = [sum(a * b for a, b in zip(q, v)) for v in emb.encode(sents)]
    best = sorted(zip(sims, sents), reverse=True)
    return " ".join(s for sc, s in best[:2] if sc >= best[0][0] - 0.15)


def grounded_answer(query, strategy=None, k=TOP_K, use_cache=True):
    strategy = strategy or deployed_strategy()
    key = (strategy, normalize_query(query))
    if use_cache and key in _CACHE:
        STATS["cache_hits"] += 1
        return {**_CACHE[key], "cache_hit": True}
    STATS["retrieval_calls"] += 1
    chunks = retrieve(query, strategy, k)
    top = chunks[0]["score"] if chunks else 0.0
    thr = get_threshold(strategy)
    if top < thr:
        res = {"answer": FALLBACK, "grounded": False, "top_score": top, "threshold": thr, "sources": [], "context": []}
    else:
        STATS["generation_calls"] += 1
        ctx = context_sentences(chunks)
        res = {"answer": mock_generate(query, ctx), "grounded": True, "top_score": top, "threshold": thr,
               "sources": sorted({c["doc_id"] for c in chunks}), "context": ctx}
    if use_cache: _CACHE[key] = res
    return {**res, "cache_hit": False}


# ---------- chunking comparison (Task 5) ----------
EVAL_SET = [("What is the return window for Footwear?", {"KB01"}), ("How many days does a COD refund take?", {"KB02"}),
            ("How long is delivery to a metro city?", {"KB03"}), ("Is my pincode eligible for reverse pickup?", {"KB04"}),
            ("What warranty comes with Electronics?", {"KB05"}), ("Can I cancel my order after it has shipped?", {"KB06"})]


def compare_strategies(verbose=True):
    ensure_index(); summary = {}
    for s in COLL:
        P, R = [], []
        if verbose: print(f"\n=== Collection: {s} (k={TOP_K} chunks -> parent docs, deduped) ===")
        for q, rel in EVAL_SET:
            docs = list(dict.fromkeys(c["doc_id"] for c in retrieve(q, s)))
            tp = len(set(docs) & rel); p = tp / len(docs); r = tp / len(rel); P.append(p); R.append(r)
            if verbose: print(f"{q}\n   retrieved={docs} relevant={sorted(rel)} TP={tp} | P={tp}/{len(docs)}={p:.2f} R={tp}/{len(rel)}={r:.2f}")
        summary[s] = {"P": sum(P) / len(P), "R": sum(R) / len(R)}
        summary[s]["F1"] = 2 * summary[s]["P"] * summary[s]["R"] / ((summary[s]["P"] + summary[s]["R"]) or 1)
        if verbose: print(f"AVG precision={summary[s]['P']:.3f} recall={summary[s]['R']:.3f} F1={summary[s]['F1']:.3f}")
    best = max(summary, key=lambda s: (round(summary[s]["F1"], 6), summary[s]["P"]))
    c = _load_calib() or calibrate(); c["strategy"] = best; c["chunk_eval"] = summary
    CALIB_FILE.write_text(json.dumps(c, indent=2))
    if verbose:
        o = "fixed" if best == "sentence" else "sentence"
        print(f"\nRECOMMENDATION: deploy '{best}' (F1 {summary[best]['F1']:.3f}, precision {summary[best]['P']:.3f}) over "
              f"'{o}' (F1 {summary[o]['F1']:.3f}, precision {summary[o]['P']:.3f}).")
    return best, summary


if __name__ == "__main__":
    print("indexed:", build_index(reset=True)); print("embedder:", get_embedder().backend)
    print(json.dumps(calibrate(), indent=1)); compare_strategies()
