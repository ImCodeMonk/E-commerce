"""Tasks 10, 15 (runtime layer) - guardrails, token budget, rate limiting."""
import re, time, threading, unicodedata
from collections import defaultdict, deque
from config import MAX_QUERY_CHARS, TOKEN_BUDGET, RATE_LIMIT_PER_MIN
from embedder import tokens

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def sanitize(text):
    """NFKC-normalise (defeats full-width digits), drop zero-width + control chars."""
    t = unicodedata.normalize("NFKC", str(text)).translate(_ZW)
    return "".join(c for c in t if c in "\n\t" or unicodedata.category(c)[0] != "C").strip()


# ---------- PII masking (fixed-format: phone + card last-4) ----------
_CARD_FULL = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_CARD_L4 = re.compile(r"(?i)((?:card|ending|last\s*4|last\s*four|xxxx)[^\d\n]{0,25})(\d{4})(?!\d)")
_PHONE = re.compile(r"(?<![\d-])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\d])")


def mask_pii(text):
    """Returns (masked_text, list_of_pii_types_found)."""
    t, found = sanitize(text), []
    t, n = _CARD_FULL.subn("[CARD]", t);  found += ["card_number"] * (n > 0)
    t, n = _CARD_L4.subn(lambda m: m.group(1) + "[CARD_LAST4]", t); found += ["card_last4"] * (n > 0)
    t, n = _PHONE.subn("[PHONE]", t); found += ["phone"] * (n > 0)
    return t, found


# ---------- prompt-injection detection ----------
_INJ = [r"ignore (?:all |any |the |your )?(?:previous|prior|above|earlier)? ?(?:instructions|rules|prompts?)",
        r"disregard (?:all |any |the |your )?(?:previous|prior|above)?",
        r"(?:reveal|show|print|leak|repeat) (?:me )?(?:your |the )?(?:system|hidden|initial) (?:prompt|instructions?)",
        r"system prompt", r"you are now", r"developer mode", r"jailbreak", r"\bdan mode\b", r"do anything now",
        r"pretend (?:to be|you are)", r"act as (?:an? )?(?:unrestricted|admin|root)", r"override (?:the )?(?:safety|rules|policy)",
        r"bypass (?:the )?(?:guardrails?|safety|filters?)", r"<\|im_(?:start|end)\|>", r"###\s*system",
        r"forget (?:everything|all|your) ", r"api[\s_-]?key", r"call (?:the )?check_order_status for (?:every|all)"]
_INJ_RE = [re.compile(p) for p in _INJ]


_LEET = str.maketrans("013457@$", "oieastas")  # matching-only copy: 'ign0re' -> 'ignore'


def detect_injection(text):
    t = " ".join(sanitize(text).lower().translate(_LEET).split())
    hits = [p.pattern for p in _INJ_RE if p.search(t)]
    return bool(hits), hits


# ---------- output-side groundedness ----------
def support_ratio(answer, context):
    a = tokens(answer)
    if not a: return 1.0
    c = set(tokens(context))
    return sum(w in c for w in a) / len(a)


def groundedness_check(answer, context_list, min_support=0.6):
    """Refuse if the answer is not supported by the retrieved context."""
    if not context_list: return False, 0.0
    r = support_ratio(answer, " ".join(context_list))
    return r >= min_support, round(r, 3)


# ---------- runtime governance: token budget ----------
class BudgetExceeded(Exception): ...


def estimate_tokens(text): return max(1, len(text) // 4)


def check_budget(text, overhead=2400, calls=5, cap=TOKEN_BUDGET):
    """Projected cost = fixed crew prompt overhead + query tokens replayed in ~5 LLM calls.
    Reject BEFORE any LLM/tool work (measured: a short query costs ~2.2k-2.9k tokens)."""
    need = overhead + estimate_tokens(text) * calls
    if need > cap:
        raise BudgetExceeded(f"projected {need} tokens > cap {cap}")
    return need


# ---------- simple per-client rate limiter ----------
_hits, _lock = defaultdict(deque), threading.Lock()


def rate_limited(client, limit=None, window=60):
    limit = limit or RATE_LIMIT_PER_MIN; now = time.time()
    with _lock:
        q = _hits[client]
        while q and now - q[0] > window: q.popleft()
        if len(q) >= limit: return True
        q.append(now); return False
