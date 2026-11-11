"""Entry point for the `project-rag` CLI: builds the services (via the facade) and runs the
argparse interface in `interfaces/cli.py`."""

from __future__ import annotations

import sys

from .api import ProjectRAG
from .interfaces.cli import run


def main(argv: list[str] | None = None) -> int:
    return run(argv, lambda config_path: ProjectRAG(config_path=config_path))


if __name__ == "__main__":
    sys.exit(main())
