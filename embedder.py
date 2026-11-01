"""Task 3 - embeddings. Real: SentenceTransformers all-MiniLM-L6-v2 (free, local).
Fallback (only if the model cannot be loaded, e.g. offline first run): deterministic hashing embedder."""
import hashlib, math, os, re

_STOP = set("a an the of to in is are was be for on at and or it its my me i can do does how what which with by from as this that".split())


def _stem(w):
    for suf in ("ies", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class HashEmbedder:
    backend = "hash-fallback"
    DIM = 1024

    def __init__(self):  # IDF learned from the KB's own sentences (deterministic, offline)
        import math as m, re as r
        from kb import KB_DOCS
        sents = [x for d in KB_DOCS for x in r.split(r"(?<=[.!?])\s+", d["text"])]
        df = {}
        for x in sents:
            for w in set(tokens(x)): df[w] = df.get(w, 0) + 1
        self.idf = {w: m.log((1 + len(sents)) / (1 + c)) + 1 for w, c in df.items()}
        self.default_idf = m.log(1 + len(sents)) + 1

    def _vec(self, text):
        v = [0.0] * self.DIM
        t = tokens(text)
        for f, wt in [(w, self.idf.get(w, self.default_idf)) for w in t] + [(a + "_" + b, 0.5) for a, b in zip(t, t[1:])]:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.DIM] += wt
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def encode(self, texts):
        return [self._vec(t) for t in texts]


class STEmbedder:
    backend = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.m.encode(list(texts), normalize_embeddings=True).tolist()


_E = None


def get_embedder():
    global _E
    if _E is None:
        if os.environ.get("EMBED_BACKEND", "").lower() == "hash":
            _E = HashEmbedder()
        else:
            try:
                _E = STEmbedder()
            except Exception as e:  # loud, never silent
                print(f"[WARN] SentenceTransformers unavailable ({type(e).__name__}); using hash fallback embedder.")
                _E = HashEmbedder()
    return _E
