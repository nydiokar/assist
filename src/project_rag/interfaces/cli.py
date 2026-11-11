"""CLI runbook for the search-only MVP (v1 WP2 gate): ingest / search / open / status.

Usage:
  project-rag ingest  --project PID --root PATH [--dry-run] [--config CFG]
  project-rag search  --project PID --query "..." [--top-k N] [--config CFG] [--root PATH]
  project-rag status  --project PID [--config CFG]
  project-rag open    --project PID --doc DOCID --source-path REL --root PATH [--config CFG]

No LLM is involved. --root is required for search/open so the server-side project root is
registered (open_source resolves paths within it). The entry point (`project_rag.cli`) supplies
`open_api`, which builds the services for a config path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from .api_protocol import AssistantAPI


def _emit(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def run(argv: list[str] | None, open_api: Callable[[str | None], AssistantAPI]) -> int:
    parser = argparse.ArgumentParser(prog="project-rag")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--project", required=True)
    p_ing.add_argument("--root", required=True)
    p_ing.add_argument("--dry-run", action="store_true")

    p_se = sub.add_parser("search")
    p_se.add_argument("--project", required=True)
    p_se.add_argument("--query", required=True)
    p_se.add_argument("--top-k", type=int, default=None)
    p_se.add_argument("--root", default=None)

    p_st = sub.add_parser("status")
    p_st.add_argument("--project", required=True)

    p_op = sub.add_parser("open")
    p_op.add_argument("--project", required=True)
    p_op.add_argument("--doc", required=True)
    p_op.add_argument("--source-path", required=True)
    p_op.add_argument("--root", required=True)

    args = parser.parse_args(argv)
    api = open_api(args.config)
    try:
        if args.command == "ingest":
            _emit(api.ingest_project(args.project, args.root, dry_run=args.dry_run).model_dump())
        elif args.command == "search":
            if args.root:
                api.register_project_root(args.project, args.root)
            _emit(api.search_project(args.project, args.query, top_k=args.top_k).model_dump())
        elif args.command == "status":
            _emit(api.get_ingestion_status(args.project).model_dump())
        elif args.command == "open":
            api.register_project_root(args.project, args.root)
            _emit(
                api.open_source(args.project, args.doc, source_path=args.source_path).model_dump()
            )
    finally:
        api.close()
    return 0
