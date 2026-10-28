# WP4 wiring options — investigation only (no wiring action taken)

**Date:** 2026-10-29 · **Status:** recommendation for the owner. **Nothing was wired.** No LLM was
called, no tool was registered, the AI-team repo was not edited (read-only inspection only).

This is the §4 deliverable of the round-2 brief: investigate the easiest convenient way to put an
LLM/agent behind the retriever, re-verify the AI-team seam against the *live* repo, and hand the
decision back. The retrieval layer is ready (`docs/HANDOFF_WP4_READY.md`): `ai_team_adapter.py`
exposes a clean `ToolDescriptor` list (`search_project`, `open_source`, `get_ingestion_status`)
with JSON schemas and a `# WP4 SEAM` marker — that is the binding point.

---

## 0. What the owner must decide (the one real fork)

Everything below reduces to **one** owner decision, because it is the only irreversible, no-clear-
default choice and it gates the no-egress rule:

> **Does corpus text (retrieved Bulgarian passages) leave this machine to a hosted LLM, or not?**
> - **No egress** → Option 1 (Ollama local) — but accept a weaker model that may give false
>   negatives on Bulgarian grounding, which the owner is explicitly worried about.
> - **Egress allowed, with a written owner decision** (spec v1 §8) → Option 2b (Claude via the
>   Agent SDK) — far stronger at Bulgarian + grounded citation + abstention, lowest effort, but
>   corpus passages go to Anthropic.

The second decision (only if egress is allowed): **which hosted model.** Recommendation below.

Everything else — *how* RAG binds as a tool — is **already solved and low-effort**, and does not
need an owner decision. See §3.

---

## 1. The three options (owner's framing), investigated

### Option 1 — Ollama local inference (no egress)

**Effort:** medium. Ollama is **not currently installed** on this machine (`which ollama` → not
found); it would need a WSL install + a model pull. It serves an OpenAI-compatible HTTP endpoint,
so our side is a thin client that passes `search_project` results as context. On AMD the ROCm build
of Ollama is required (the card is a RX 7900 XT / gfx1100, same device the retriever already uses).

**Does it work / is it a *fair* test?** This is the owner's own worry, and it is the real risk.
A local model that confabulates or mangles Cyrillic would produce **false negatives** — the demo
would look like RAG failed when the *retrieval* (measured recall@5 0.82) was fine. So the bar is:
a locally-runnable model that is genuinely strong at **Bulgarian + grounded citation + abstention**.

- **VRAM budget (measured, hard constraint):** the card is **21.4 GB** total; the retriever already
  peaks at **10.7 GB** (Qwen3-4B embedder + bge reranker, `docs/EVAL_LOG.md` 2026-10-28). That
  leaves **~10.7 GB** for a co-resident LLM. That fits a **7–9B model at 4-bit** (~5–6 GB) with
  headroom, but **not** a 30B+ model co-resident. Options to relieve this: (a) run the LLM 4-bit and
  co-resident (simplest), or (b) swap — unload the reranker between retrieve and generate (adds
  latency, more code), or (c) run the LLM on CPU (slow, but the generate step is not latency-critical
  for a demo).
- **Bulgarian-capable local candidates (to be *measured*, not assumed — same discipline as the rest
  of this project):** multilingual instruct models in the 7–9B class that fit the budget — e.g.
  `qwen2.5:7b-instruct` (Qwen family already trusted here for Cyrillic tokenization), `gemma2:9b`,
  `aya-expanse:8b` (Cohere, explicitly multilingual incl. Bulgarian). **Do not pick one from
  reputation — run the owner-eval (`scripts/ask.py` surface + the report questions, Task D) through
  each candidate and read the Bulgarian answers.** A local model is only acceptable if it (i) keeps
  Cyrillic intact, (ii) cites the retrieved passage rather than its own prior, and (iii) abstains
  when the retriever abstains (our gate already hands it an explicit "insufficient evidence"
  verdict — the model must honor it, not override it).
- **Egress:** none. Keeps the v1 §8 rule clean with no owner decision needed.

**Honest verdict:** the *right* choice if no-egress is non-negotiable, but it front-loads
model-selection risk and a measurement round. It is NOT the lowest-effort path and it carries the
demo-invalidation risk the owner named.

### Option 2 — reuse the AI-team harness (RAG as a tool inside it)

**Re-verified against the live repo 2026-10-29 (read-only; the 2026-10-16 note is now updated).**
The seam has **moved on and is more favorable than the old note said:**

- AI-team's Claude backend (`src/backends/claude_driver.py`) now builds sessions with the
  **Claude Agent SDK** (`claude_agent_sdk.ClaudeAgentOptions` / `ClaudeSDKClient`), not a raw CLI
  shell-out. `_sdk_options()` (claude_driver.py:973) is where a session's tools and MCP servers are
  assembled.
- **There is already a working in-process SDK-MCP-tool injection pattern to copy.**
  `src/control/agent_sender.py:build_claude_sender_server()` does exactly this for an internal
  `send_instruction` tool:
  ```python
  from claude_agent_sdk import create_sdk_mcp_server, tool

  @tool(SEND_TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA)
  async def _send(args): ...
  return dict(create_sdk_mcp_server(name=SENDER_SERVER_NAME, version="1.0.0", tools=[_send]))
  ```
  and `_sdk_options()` merges it via `mcp_servers={SENDER_SERVER_NAME: build_claude_sender_server(slot)}`
  plus adding the tool FQN (`mcp__<server>__<tool>`) to `allowed_tools`. **This is a drop-in template
  for a RAG MCP server** — our `ai_team_adapter.ToolDescriptor` list already has the names, JSON
  schemas, and handlers; wrapping each as an `@tool` and `create_sdk_mcp_server(...)` is a
  near-mechanical translation of a pattern AI-team already runs in production.
- **Still true (confirmed live):** `permission_mode="bypassPermissions"` is hard-coded
  (claude_driver.py:996) — built-in tools (the agent's own bash/read/write) are **not** disableable
  per session without changing AI-team code. For a grounded-answer guarantee you want the agent to
  answer *only* from `search_project`, so this matters: either accept that the agent *can* read the
  filesystem directly (it shouldn't need to — the corpus isn't in its cwd), or land a small AI-team
  change to allow a restricted tool set. This is an AI-team-side change (out of scope here).

**Egress:** AI-team's Claude backend talks to **hosted Claude** — so this is an egress path and
needs the same owner decision as 2b. (AI-team also wraps Codex/opencode; same egress consideration.)

**Effort:** medium-high. The tool binding is low-effort (copy the sender pattern), but you inherit
AI-team's whole session/worker/gateway machinery, its `bypassPermissions` default, and its strict
test-cost rule (its tests can invoke the **paid** Claude CLI — never run them). Worth it only if the
owner actually wants the full agent harness (multi-turn, tool-use loop, its UI) around the RAG, not
just "an LLM that answers with citations."

### Option 2b — direct Claude via the Claude Agent SDK / Messages API (the lowest-effort path that works)

Skip the AI-team harness; call Claude directly from a small script that sits next to
`scripts/ask.py`. Two shapes, both use the official `anthropic` SDK (loaded the `claude-api` skill
for this — do not hand-roll):

1. **Plain Messages API (simplest).** For each owner question: run `search_project` → format the
   retrieved passages + the evidence verdict into the prompt → one `client.messages.create(...)`
   with a grounding system prompt → stream the grounded answer with citations. The abstention gate
   already returns "insufficient evidence" when it should; the system prompt tells Claude to say
   "Not established from accessible project documents" (v1 §5.2) when the verdict is insufficient or
   the passages don't support an answer. **This is retrieval-first by construction** — the retriever
   runs first, the LLM only grounds. No tool-use loop needed.
2. **Tool-use loop (Agent SDK or Messages tool-use).** Register the three `ai_team_adapter` tools as
   `anthropic` tools and let Claude call `search_project`/`open_source` itself. More capable
   (multi-hop, self-directed), slightly more code (the tool-call loop), and it mirrors exactly what
   Option 2 does inside AI-team — so if you build 2b's tool loop cleanly it is portable into AI-team
   later.

**Model choice (per the `claude-api` skill; this is the second owner decision, and only if egress is
allowed):** default to **`claude-opus-4-8`** for the demo — strongest Bulgarian + grounded reasoning
+ honest abstention, which is exactly the fitness the owner is worried a weak local model would fail.
If cost matters more than ceiling, **`claude-sonnet-4-6`** is the balanced choice and is still far
above any 7–9B local model on Bulgarian grounding. (Do not use `budget_tokens` — these models use
adaptive thinking; the skill has the exact call shape.) **Egress implication:** the retrieved
Bulgarian corpus passages are sent to Anthropic — this triggers the v1 §8 no-egress rule and needs
the owner's explicit written go-ahead before any call is made.

**Effort:** **lowest of the three.** ~1 small script, the `anthropic` SDK, a grounding prompt, reuse
of the existing warm `ProjectRAG`. No AI-team dependency, no Ollama install, no local-model
measurement round.

---

## 2. Comparison

| | Option 1 — Ollama local | Option 2 — AI-team harness | Option 2b — direct Claude SDK |
|---|---|---|---|
| **Effort** | Medium (install + model pull + measure a BG-capable model) | Medium-high (inherit harness + `bypassPermissions` + its test-cost rule) | **Lowest** (one script + `anthropic` SDK) |
| **Does-it-work confidence** | Risky — depends on a 7–9B model being good enough at BG grounding; owner's false-negative worry is real | High (hosted Claude), but heavy to stand up | **High** (hosted Claude) and simple |
| **Egress / privacy** | **None** (clean v1 §8) | Hosted Claude → **egress**, owner decision | Hosted Claude → **egress**, owner decision |
| **Bulgarian + grounding + abstention fitness** | Must be *measured*; marginal at this VRAM budget | Strong | **Strongest** (Opus 4.8 / Sonnet 4.6) |
| **RAG binds as** | HTTP client passes passages as context (no tool loop) | **MCP tool** — copy `agent_sender` SDK-MCP pattern into `_sdk_options` | **built-in tool** (Messages tool-use) *or* passages-as-context (plain) |
| **VRAM** | Co-resident must fit in ~10.7 GB free of 21.4 GB → 4-bit 7–9B only | N/A (hosted) | N/A (hosted) |

---

## 3. The binding is NOT the hard part (settled, no owner decision)

Whichever option is chosen, *how RAG attaches as a tool* is already low-effort and does not need a
decision:

- **As an MCP tool (Option 2, and Option 2b tool-loop):** `ai_team_adapter.build_tool_descriptors()`
  already returns typed handlers + JSON schemas for `search_project` / `open_source` /
  `get_ingestion_status`. Wrapping each in the Agent SDK's `@tool` + `create_sdk_mcp_server(...)` is a
  near-mechanical copy of the **working** `agent_sender.build_claude_sender_server` pattern verified
  live in AI-team today. The `# WP4 SEAM` marker in `ai_team_adapter.register_with_ai_team` is where
  that binding lands.
- **As passages-as-context (Option 1, and Option 2b plain):** no tool registration at all — just call
  `search_project`, format `results[]` + `evidence` verdict into the prompt. Simplest possible path.

The retriever already emits everything a grounded, abstaining answerer needs: citable
`source_path`/location per result, the `evidence` sufficiency verdict, parent-section
`expanded_context` (Task A), and `doc_class` filtering to avoid reference-material bleed (the Task D
finding). WP4 inherits a ready contract.

---

## 4. Recommended path

**Recommend Option 2b (plain Messages API first), gated on the owner's egress decision.** Reasons:

1. **Lowest effort, highest does-it-work confidence** — the demo-level calibration the brief asks
   for. One script beside `scripts/ask.py`, reusing the warm `ProjectRAG`.
2. **Best fitness for the exact worry** — Opus 4.8 / Sonnet 4.6 are strong at Bulgarian + grounded
   citation + honest abstention; a weak local model is the owner's stated false-negative risk.
3. **Retrieval-first by construction** — retriever runs, verdict gates, LLM only grounds; no tool
   loop needed for the first demo. The tool-loop shape (2b variant 2) is the clean upgrade and is
   portable into AI-team (Option 2) later if the owner wants the full harness.

**But Option 2b and Option 2 both send corpus text to Anthropic.** If the owner will not allow any
egress of corpus content, the recommendation flips to **Option 1 (Ollama local)** — and the next
step there is a *measured* model bake-off (run `scripts/ask.py` + the Task D report questions through
2–3 BG-capable 7–9B models and read the Bulgarian answers), not picking a model by reputation.

### The single decision to hand back to the owner

> **May retrieved Bulgarian corpus passages be sent to a hosted LLM (Anthropic) for the demo?**
> - **Yes** → proceed with Option 2b; default model **`claude-opus-4-8`** (or `claude-sonnet-4-6`
>   for cost), record the egress decision in `docs/DECISIONS.md` per v1 §8 before any call.
> - **No** → Option 1 (Ollama local); first task is a measured BG-grounding bake-off within the
>   ~10.7 GB free VRAM budget.

Either way: **no wiring is done until the owner answers.** This document is the stop point.

---

## 5. Appendix — live AI-team seam facts (re-verified 2026-10-29, read-only)

Supersedes the 2026-10-16 note in `AGENTS.md` where it differs. Read AI-team's `CLAUDE.md` before
running anything there — its tests can invoke the **paid** Claude CLI; they were **not** run here.

- `src/backends/claude_driver.py:_sdk_options()` (line 973) builds `ClaudeAgentOptions` from the
  **Claude Agent SDK** (`claude_agent_sdk`). MCP servers merge in via `mcp_servers={...}`; tool FQNs
  (`mcp__<server>__<tool>`) are appended to `allowed_tools`.
- `src/control/agent_sender.py:build_claude_sender_server()` (line 215) is a **working in-process
  SDK-MCP server** using `create_sdk_mcp_server` + `@tool` — the template for a RAG MCP server.
- `permission_mode="bypassPermissions"` is hard-coded (claude_driver.py:996) — built-in agent tools
  are not per-session disableable without an AI-team code change. Relevant if you want the agent to
  answer *only* from RAG tools.
- MCP servers remain mergeable per session in-process (the secret/slot stays in carrier memory — no
  CLI env/argv/config file). The old note's "host-global `~/.claude.json` only" framing is outdated:
  in-process per-session SDK MCP injection is the live mechanism.
- Not audited this round (flagged, not blocking): whether tool args/results are now persisted in
  `llm_events` (telemetry lives in `src/control/telemetry_store.py` / `telemetry_sink.py` /
  `src/core/telemetry_adapters/claude_stream_json.py` — inspect there if provenance logging of RAG
  tool calls is required for the grounding gate).
