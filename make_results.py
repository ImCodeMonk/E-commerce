"""Generates artifacts/RESULTS.md (live numbers for the README). Run AFTER installing sentence-transformers."""
import config  # noqa: F401
import json, time, warnings
warnings.filterwarnings("ignore")
import dataset, rag, tools
from embedder import get_embedder

rag.build_index(reset=True); cal = rag.calibrate(); best, summ = rag.compare_strategies(verbose=False)
cc, sc, pct = dataset.stats(dataset.ORDERS)
from evaluate import run_eval
rows, avg = run_eval(verbose=False)
L = [f"# Measured results (embedder: `{get_embedder().backend}`)", "",
     "## Dataset design (Task 1)", f"- seed={dataset.SEED} (seed 42 gave 8.33% delayed, outside the 10-30% band, so the seed was changed - no record was hand-edited)",
     f"- n={len(dataset.ORDERS)}; category weights {dict(zip(dataset.CATEGORIES, dataset.CATEGORY_WEIGHTS))}; status weights {dict(zip(dataset.STATUSES, dataset.STATUS_WEIGHTS))}",
     f"- amount range (INR) per category: {dataset.PRICE_RANGE} (typical Nykaa basket sizes: low for beauty/apparel, high for electronics)",
     f"- counts per category {cc}; per status {sc}; delayed_shipment = {pct}%", "",
     "## Threshold calibration (Task 4)"]
for s in rag.COLL:
    m = cal["measured"][s]; L.append(f"- `{s}`: in-scope top-1 {m['in_scope']} | out-of-scope top-1 {m['out_of_scope']} -> **threshold {cal['threshold'][s]}**")
L += ["", "## Chunking comparison (Task 5)"] + [f"- `{s}`: precision {summ[s]['P']:.3f}, recall {summ[s]['R']:.3f}, F1 {summ[s]['F1']:.3f}" for s in summ]
o = "fixed" if best == "sentence" else "sentence"
L += [f"- **Recommendation:** deploy `{best}` (F1 {summ[best]['F1']:.3f}) over `{o}` (F1 {summ[o]['F1']:.3f}); recall is the safety-critical number for a support bot and precision is bounded because top-3 chunks map to 2-3 parent docs.", "",
      "## Escalation score (Task 6)", "- `score = 0.6*delayed_shipment + 0.4*(days_since_created/30)`",
      f"- threshold = **{tools.ESCALATION_THRESHOLD}** = 80th percentile of the {len(tools.SCORES)} scores in my dataset (flags {sum(s >= tools.ESCALATION_THRESHOLD for s in tools.SCORES)} orders)", "",
      "## Evaluation averages over 15 queries (Task 13)", f"- {avg}", "", "| # | Acc | Grd | Cmp | Saf | query |", "|--|--|--|--|--|--|"]
L += [f"| {i} | {s['accuracy']} | {s['grounding']} | {s['completeness']} | {s['safety']} | {q} |" for i, (q, r, s) in enumerate(rows, 1)]
(config.ART / "RESULTS.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
