# Nykaa Domain Support Agent (CrewAI) — Final Capstone

**Track completed: Nykaa (E-commerce & Retail).** Submitted for IIT Roorkee · Author: `<your name / enrolment no.>`

Everything runs under **`MOCK_LLM`** (zero API keys, zero LLM network calls). `CREWAI_DISABLE_TELEMETRY=true` is set
automatically in `config.py` before CrewAI is imported (and `OTEL_SDK_DISABLED`, `ANONYMIZED_TELEMETRY=False` for Chroma).
The only one-time download is the free `all-MiniLM-L6-v2` embedding model (~90 MB); after that set `HF_HUB_OFFLINE=1`.

## Dataset-design choices (Part 1, Task 1 — reproducible)
- `seed=7` (seed 42 produced 8.33% delayed, outside 10–30%, so the **seed was changed**; no record was hand-edited), `n=60`
- category weights Apparel .25 / Electronics .15 / Home .15 / Footwear .20 / Beauty .25
- status weights Placed .15 / Shipped .25 / Delivered .35 / Returned .15 / Refunded .10; `delayed_shipment` ~ Bernoulli(0.20)
- amount (INR): Apparel 499–4999, Electronics 1999–59999, Home 799–14999, Footwear 999–7999, Beauty 199–3999
  (realistic Nykaa basket sizes: small for beauty/apparel, large for electronics)
- Result: 60 orders, every category ≥ 8, every status ≥ 7, delayed = 18.33%. `python dataset.py` prints and asserts this.

## Run in VS Code (Windows / macOS / Linux, Python 3.10–3.12)
```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python run_all.py                # runs Tasks 1-16 in order and prints the transcript
python make_results.py           # writes artifacts/RESULTS.md with YOUR measured numbers
python -m pytest tests -q        # 48 tests incl. security edge cases
uvicorn app:app --port 8000      # API docs at http://127.0.0.1:8000/docs
```
Save transcripts: `python run_all.py > transcripts/full_run.txt` (PowerShell: `python run_all.py | Tee-Object transcripts/full_run.txt`).

> **Paste the numbers from `artifacts/RESULTS.md`** (calibration values, threshold, precision/recall, recommendation, 15-query
> scores) into this README after running on your machine — they depend on the embedding model, so they are generated, never hard-coded.
> If the model cannot be downloaded, `embedder.py` prints a loud `[WARN]` and uses a deterministic offline fallback (results then differ).

## Where each task lives
| Task | File | Evidence |
|---|---|---|
| 1 dataset | `dataset.py` | `python dataset.py` |
| 2 KB (12 docs) | `kb.py` | tests |
| 3 two chunkers → 2 Chroma collections | `rag.py` (`chunk_fixed`, `split_sentences`, `upsert`) | run_all T3 |
| 4 grounded generation + calibrated threshold | `rag.py` (`calibrate`, `grounded_answer`) | run_all T4 |
| 5 precision/recall per collection | `rag.py` (`compare_strategies`) | per-query arithmetic printed |
| 6 `check_order_status` + escalation | `tools.py` | formula + 80th-percentile threshold |
| 7 CrewAI crew (3 agents) + `MockLLM` | `crew.py`, `mock_llm.py` (extends `BaseLLM`) | tool-call log |
| 8 session memory | `memory.py` (`RunnableWithMessageHistory`) | two transcripts |
| 9 structured output | `crew.py` `CrewResponse` (`output_pydantic` + explicit `model_validate_json`) | T9 |
| 10 guardrails | `security.py` | T10 |
| 11 FastAPI + WebSocket | `app.py` | T11, tests |
| 12 JSON-Lines logs | `app.py` `write_log` → `logs/requests.jsonl` | PII-free assert |
| 13 LLM-as-judge ×15 | `evaluate.py` | table + 4 averages |
| 14 Autogen review | `review.py` (`RoundRobinGroupChat(max_turns=2)`, `output_content_type=Verdict`, `custom_message_types`) | approve + revise |
| 15 governance | `tools.py` (least autonomy), `security.py` (budget), `pipeline.py` | T15 |
| 16 cache | `rag.py` (`_CACHE`, `STATS`) | before/after counters |

## Design notes
- **Escalation:** `score = 0.6·delayed + 0.4·(days/30)`; threshold = 80th percentile of the dataset's own scores.
- **Pitfalls handled in `mock_llm.py`:** tool results are parsed only from the model's own assistant message (never the system
  template that contains `Observation: the result of the action`); tool arguments are chosen from the tool's declared `required` schema, not its name.
- **Risk level: MEDIUM** (customer-support tickets class). It reads order status/value but no medical/hiring/credential data, PII is masked
  before model and logs, answers are grounded + independently reviewed, and the system can't change any state (no refunds/cancellations).
- **Least autonomy:** `TOOL_PERMISSIONS` allows `check_order_status` only for `Lookup Agent`; `make_agent` raises `PermissionError` otherwise (demo + test).
- **Runtime budget:** projected tokens are checked *before* any LLM/tool work (HTTP 413), plus a hard ceiling while running.
- **Caching:** keyed by normalised query text; invalidated whenever a document is added.

## Security checklist implemented
Unicode NFKC + zero-width/control-char stripping (defeats full-width-digit and invisible-char bypasses) · PII masking of phone and card
(full/last-4) on input, in memory history, in logs and on output · leetspeak-aware prompt-injection detection · forged-verdict-token
resistance in the review stage · order-id regex whitelist (blocks SQL/path-style input) · input length cap + token budget · per-client rate limit ·
`/add-document`: optional `X-API-Key` (constant-time compare), 2–5 sentence rule, PII rejection, injection (KB-poisoning) rejection,
reserved `KBnn` ids can't be overwritten · session/memory size caps · generic 500 errors (no stack-trace leaks) · WebSocket survives
disconnects, bad frames and oversized messages. Customer name/address are not masked (out of scope per brief); use fabricated data only.

## Known limits (honest)
Mock LLM answers are extractive, not generative; masking covers fixed-format fields only; the in-process cache, memory and rate limiter
are per-process (single worker). Added documents persist in `chroma_db/`; `build_index(reset=True)` restores the base KB.
