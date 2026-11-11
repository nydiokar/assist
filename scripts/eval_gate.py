"""`make eval-gate` (spec §6.8.1, A18): score the gold set, compare with the last promoted
baseline under the tolerance policy, and promote on a pass. Runs on the host or the PC replica,
never in CI (the gold set quotes the real corpus).

  python scripts/eval_gate.py --config configs/base.yaml --project youth-participation \
      --root /corpus/"Youth Participation" --questions eval/yp_questions_v3.jsonl \
      --state eval/results/gate_state.json

Gate state file (local, gitignored): {"baseline": GateMetrics, "floor": GateMetrics | null,
"pipeline": fingerprint}. The floor is set once by the owner (`--set-floor-from-baseline` writes
the E1 proposal = baseline - 3 questions) and never moves automatically. A question whose
`expect_source_hashes` no longer match the corpus is stale and excluded (§6.8.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from project_rag.api import ProjectRAG  # noqa: E402
from project_rag.core.fingerprint import pipeline_fingerprint  # noqa: E402
from project_rag.core.gate import GateMetrics, GatePolicy, gate, propose_floor  # noqa: E402
from project_rag.core.smoke import stale_questions  # noqa: E402
from project_rag.services.eval import evaluate, load_questions, metrics_for_gate  # noqa: E402
from project_rag.wiring import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--state", default="eval/results/gate_state.json")
    ap.add_argument("--set-floor-from-baseline", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rag = ProjectRAG(config=cfg)
    rag.register_project_root(args.project, args.root)
    questions = load_questions(args.questions)
    gen = rag.services.catalog.active_generation(args.project)
    current = {
        d.source_path: d.content_hash
        for d in (rag.services.catalog.document_states(args.project, gen) if gen else [])
    }
    labelled = {q.id: q.expect_source_hashes for q in questions if q.expect_source_hashes}
    stale = set(stale_questions(labelled, current))
    fresh = [q for q in questions if q.id not in stale]
    report = evaluate(rag, args.project, fresh, top_k=5)
    candidate = metrics_for_gate(report)
    fingerprint = (
        pipeline_fingerprint(
            cfg.parsing, cfg.chunking, cfg.classification, cfg.embedding.profile().key()
        )
        + f"|retrieval={cfg.retrieval.model_dump_json()}"
    )

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    out: dict[str, object] = {"candidate": candidate.model_dump(), "stale": sorted(stale)}
    if "baseline" not in state:
        state = {"baseline": candidate.model_dump(), "floor": None, "pipeline": fingerprint}
        out["decision"] = "first run: baseline recorded (not gated)"
    else:
        baseline = GateMetrics.model_validate(state["baseline"])
        floor = GateMetrics.model_validate(state["floor"]) if state.get("floor") else None
        decision = gate(
            candidate,
            baseline,
            GatePolicy(floor=floor),
            pipeline_changed=state.get("pipeline") != fingerprint,
        )
        out["decision"] = decision.model_dump()
        if decision.passed:
            state["baseline"] = decision.baseline.model_dump()
            state["pipeline"] = fingerprint
        if not decision.passed:
            print(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit(1)
    if args.set_floor_from_baseline:
        state["floor"] = propose_floor(GateMetrics.model_validate(state["baseline"])).model_dump()
        out["floor_set"] = state["floor"]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    rag.close()


if __name__ == "__main__":
    main()
