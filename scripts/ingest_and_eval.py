r"""Ingest the YP corpus with a given config then run retrieval evaluation. (A02 pilot-gate script)

Designed to be launched as a watched/detached job:
  - All output is line-buffered stdout (captured by the job runner's log).
  - Ingest is resumable: the embedding cache (F13) commits every batch; re-running this script
    after an interruption skips already-embedded chunks and resumes from the next batch.
  - Abort criteria (A02): any non-finite vector, RSS > 24 GB, throughput < 60 tok/s detected
    inside Qwen3Embedder._validate_matrix; ingest will raise before writing bad vectors.
  - Progress: each source file prints a one-line status as it is staged. Eval prints per-question
    hit/miss. Final JSON written to eval/results/<run_name>.json.

Usage (PowerShell):
  $env:PYTHONPATH="src"
  .\.venv\Scripts\python.exe -X utf8 scripts\ingest_and_eval.py `
      --config configs\yp_qwen4b_gpu.yaml `
      --project yp_gpu `
      --root yspdb `
      --questions eval/yp_questions.jsonl `
      --run-name yp_qwen4b_gpu

Arguments:
  --config       Path to retrieval.yaml variant.
  --project      Project ID (must be a fresh name if the config profile changed).
  --root         Path to the corpus root directory.
  --questions    Path to the .jsonl eval questions file.
  --oos-questions  Optional second .jsonl (out-of-scope questions).
  --run-name     Label for the output JSON file.
  --channels     Space-separated list of channel sets to evaluate.
                 Default: lexical dense lexical+dense
  --skip-ingest  Skip ingest (use an already-indexed project). Useful for re-running eval only.
  --skip-eval    Run ingest only (no eval pass).
  --validate-labels  Validate that every expected substring exists in parsed source text before
                     running eval. Default: on. Disable with --no-validate-labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---- helpers ----------------------------------------------------------------


def log(msg: str) -> None:
    """Line-buffered timestamped print — captured by the job log verbatim."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _check_imports() -> None:
    """Fail early with a clear message if sentence-transformers / torch are missing."""
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        log(f"ABORT: missing dependency — {e}")
        log("Install with:  .venv\\Scripts\\python.exe -m pip install -e '.[qwen3]'")
        sys.exit(1)


# ---- ingest -----------------------------------------------------------------


def run_ingest(config_path: str, project: str, root: str) -> None:
    from project_rag.api import ProjectRAG
    from project_rag.domain.models import DocState

    log(f"=== INGEST  config={config_path}  project={project}  root={root}")
    rag = ProjectRAG(config_path=config_path)

    cfg = rag.config
    problems = cfg.assert_release_safe()
    if problems:
        log("ABORT: release-safety check failed:")
        for p in problems:
            log(f"  {p}")
        rag.close()
        sys.exit(1)

    log(
        f"embedding.provider={cfg.embedding.provider}  "
        f"model={cfg.embedding.qwen3.model if cfg.embedding.provider == 'qwen3' else '?'}  "
        f"device={cfg.embedding.qwen3.device if cfg.embedding.provider == 'qwen3' else 'n/a'}  "
        f"batch_size={cfg.embedding.qwen3.batch_size if cfg.embedding.provider == 'qwen3' else '?'}"
    )

    t0 = time.time()
    try:
        report = rag.ingest_project(project, root)
    except Exception as e:
        log(f"ABORT: ingest raised: {e}")
        rag.close()
        raise

    elapsed = time.time() - t0
    log(f"ingest done in {elapsed:.0f}s — gen={report.index_generation}")
    log(
        f"  added={report.added} modified={report.modified} "
        f"unchanged={report.unchanged} deleted={report.deleted}"
    )
    if report.warnings:
        for w in report.warnings:
            log(f"  WARN: {w}")

    ready = sum(1 for f in report.files if f.state == DocState.READY)
    not_ready = [
        (f.source_path, f.state, f.diagnostics) for f in report.files if f.state != DocState.READY
    ]
    log(f"  files READY={ready}  non-ready={len(not_ready)}")
    for sp, st, diags in not_ready[:30]:  # cap to avoid log spam
        log(f"    {st.value:20s} {sp}  {'; '.join(diags)[:120]}")
    if len(not_ready) > 30:
        log(f"    ... and {len(not_ready) - 30} more non-ready files (see full results JSON)")

    truncated = [
        (f.source_path, f.diagnostics)
        for f in report.files
        if any("TRUNCATION" in d for d in f.diagnostics)
    ]
    if truncated:
        log(f"  TRUNCATION in {len(truncated)} file(s) — review chunking config:")
        for sp, diags in truncated[:10]:
            log(f"    {sp}: {'; '.join(d for d in diags if 'TRUNCATION' in d)}")

    n_chunks = sum(f.n_chunks for f in report.files)
    log(f"  total chunks indexed: {n_chunks}")

    status = rag.get_ingestion_status(project)
    log(
        f"  status: n_documents={status.n_documents} n_chunks={status.n_chunks} "
        f"healthy={status.healthy}"
    )
    rag.close()


# ---- label validation -------------------------------------------------------


def validate_labels(questions: list, root: Path) -> None:
    from project_rag.adapters.corpus_local_fs import _long_path
    from project_rag.adapters.parse_registry import ParserRegistry
    from project_rag.wiring import load_config

    reg = ParserRegistry(load_config(None).parsing)
    cache: dict[str, str] = {}
    problems: list[str] = []
    for q in questions:
        for sp in q.expect_source_paths:
            if sp not in cache:
                full = _long_path(str((root / sp).resolve()))
                doc = reg.parse(full, sp, Path(sp).suffix.lower())
                cache[sp] = "\n".join(b.text for b in doc.blocks)
        for sub in q.expect_substrings:
            if not any(sub in cache.get(sp, "") for sp in q.expect_source_paths):
                problems.append(f"{q.id}: substring not in parsed source: {sub!r}")
    if problems:
        log(f"LABEL PROBLEMS ({len(problems)}):")
        for p in problems:
            log(f"  {p}")
        raise SystemExit(1)
    log(f"labels validated: {len(questions)} questions OK")


# ---- eval -------------------------------------------------------------------


def run_eval(
    config_path: str,
    project: str,
    root: str,
    questions_path: str,
    oos_path: str | None,
    channels: list[str],
    run_name: str,
    do_validate: bool,
) -> dict:
    from project_rag.api import ProjectRAG
    from project_rag.services.eval import evaluate, load_questions

    all_results: dict[str, object] = {}
    out_path = Path("eval/results") / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_question_set(label: str, qpath: str) -> None:
        qs = load_questions(qpath)
        log(f"=== EVAL  label={label}  questions={len(qs)}  channels={channels}")
        if do_validate:
            validate_labels(qs, Path(root))

        rag = ProjectRAG(config_path=config_path)
        rag.register_project_root(project, root)
        try:
            for cs in channels:
                t0 = time.time()
                rep = evaluate(rag, project, qs, top_k=5, channels=cs.split("+"))
                elapsed = time.time() - t0
                key = f"{label}/{cs}"
                all_results[key] = rep.model_dump()
                log(
                    f"\n[{key}] Recall@5={rep.recall_at_5:.3f}  "
                    f"evidence@5={rep.evidence_recall_at_5:.3f}  "
                    f"MRR={rep.mrr:.3f}  n={rep.n}  elapsed={elapsed:.1f}s"
                )
                for r in rep.per_question:
                    hit = "HIT " if r.hit_at_5 else "miss"
                    ev = "EV" if r.evidence_hit_at_5 else "--"
                    log(f"  {r.id:6s} {r.kind:18s} {hit} {ev}  rr={r.reciprocal_rank:.2f}")
        finally:
            rag.close()

    run_question_set("yp", questions_path)
    if oos_path and Path(oos_path).is_file():
        run_question_set("oos", oos_path)

    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"results written → {out_path}")
    return all_results


# ---- summary ----------------------------------------------------------------


def print_summary(results: dict) -> None:
    log("\n=== SUMMARY")
    for key, rep_raw in results.items():
        rep = rep_raw if isinstance(rep_raw, dict) else {}
        log(
            f"  {key:35s}  "
            f"Recall@5={rep.get('recall_at_5', '?'):.3f}  "
            f"evidence@5={rep.get('evidence_recall_at_5', '?'):.3f}  "
            f"MRR={rep.get('mrr', '?'):.3f}  "
            f"n={rep.get('n', '?')}"
        )


# ---- main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="yp_gpu")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_questions.jsonl")
    ap.add_argument("--oos-questions", default="eval/yp_questions_oos.jsonl")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--channels", nargs="*", default=["lexical", "dense", "lexical+dense"])
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--validate-labels", dest="validate_labels", action="store_true", default=True)
    ap.add_argument("--no-validate-labels", dest="validate_labels", action="store_false")
    args = ap.parse_args()

    log("=== ingest_and_eval.py start")
    log(f"config={args.config}  project={args.project}  root={args.root}")
    log(f"run_name={args.run_name}  channels={args.channels}")

    if not args.skip_ingest:
        _check_imports()
        run_ingest(args.config, args.project, args.root)

    if not args.skip_eval:
        results = run_eval(
            config_path=args.config,
            project=args.project,
            root=args.root,
            questions_path=args.questions,
            oos_path=args.oos_questions,
            channels=args.channels,
            run_name=args.run_name,
            do_validate=args.validate_labels,
        )
        print_summary(results)

    log("=== ingest_and_eval.py done")


if __name__ == "__main__":
    main()
