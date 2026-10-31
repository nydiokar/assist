#!/usr/bin/env bash
# Launch the YP project-assistant as a LOCKED interactive Claude chat for hand-off.
#
# What the lock guarantees (so the agent cannot cheat by reading raw files and
# burning tokens, pretending it used RAG):
#   --allowed-tools   : ONLY the three project-rag MCP tools are callable.
#   --disallowed-tools: Read/Write/Edit/Bash/Glob/Grep/Web* are STRIPPED — the agent
#                       physically cannot open the 3.5 GB corpus, so any grounded
#                       answer came through retrieval. Verified 2026-10-31: with these
#                       flags the agent reports "Read tool is disabled, I cannot access
#                       the directory" and answers only via search_project.
#   system prompt     : retrieval-first + abstention + cite source_path.
#
# Run FROM the assist dir so .mcp.json (project-local project-rag) loads:
#   cd /mnt/c/Users/Cicada38/Projects/assist && bash scripts/demo_assistant.sh
# The MCP server is spawned on demand by claude and reaped on exit — nothing to start/stop.
# First question is ~30 s (cold model load in WSL); then warm for the session.
set -euo pipefail

MODEL="${MODEL:-claude-sonnet-4-5}"   # sonnet: Haiku mis-reads the budget total (ledger F21)

SYS="You are the Youth Participation (Младежко участие) project assistant. The corpus is one \
completed Erasmus+ KA154 youth project, mostly Bulgarian with some English. You have NO file \
access — your ONLY way to see the documents is the project-rag retrieval tools. Rules: (1) For \
every question about the project, call search_project FIRST; rephrase and search again if the \
first passages do not clearly contain the asked fact. (2) Ground every claim in a returned \
passage and cite its source_path. (3) Never do arithmetic across passages and never treat a \
topically-related passage as the answer; read the value a passage states verbatim. (4) If no \
passage states the asked fact, reply exactly 'Not established from accessible project documents.' \
(5) Answer in the language of the question; preserve Bulgarian (Cyrillic) exactly."

exec claude --model "$MODEL" \
  --allowed-tools \
    "mcp__project-rag__search_project" \
    "mcp__project-rag__open_source" \
    "mcp__project-rag__get_ingestion_status" \
  --disallowed-tools \
    "Read" "Write" "Edit" "Bash" "Glob" "Grep" "WebFetch" "WebSearch" "NotebookEdit" \
  --append-system-prompt "$SYS"
