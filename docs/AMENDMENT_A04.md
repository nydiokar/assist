# Amendment A04 — S6 load check restated to match spec §6.3 backpressure

**Applies to:** `docs/V3_BUILD_PLAN.md` S6 exit checkpoint, bullet 2 (the load test).
**Status:** adopted 2026-11-13 by the agent, on the precedence rule (A03 > plan). The owner can
reverse it (see Reversibility).
**Touches:** `tests/integration/test_service_http.py::test_concurrent_searches_during_ingest`,
the busy response body in `interfaces/http/app.py`. Spec text unchanged.

## What changes

The S6 load check read: "20 concurrent `search_project` calls **during** an ingest → no errors, no
SQLite thread errors, p95 recorded. The 17th waiting request gets a 503." It now reads:

> 20 concurrent `search_project` calls during an ingest, from clients that honour `Retry-After`
> once → **no failed call**, no SQLite thread errors, p95 recorded, first-try refusals reported.
> With ≥ 17 calls waiting, the next one gets HTTP 503 + `Retry-After`, with a body MCP clients
> show as "assist is busy … retry".

## Why (measured)

- Spec A03 §6.3 fixes the behaviour: one store thread, "the request queue is bounded (≤ 16
  waiting, then a 503 with retry-after)". The plan's two bullets cannot both hold for a true
  20-call burst: 20 MCP clients started together put 17–18 calls in flight (each store call ~11
  ms), so 1–3 are refused on the first try. That is the specified behaviour, not an error.
- Measured 2026-11-13 (12 runs, S6 worktree, fixture corpus, ingest running): p95 3.26–5.43 s,
  max ≤ 5.55 s, `busy_first_try` 2–3, **0 failed calls** after one retry.
- While measuring, the test was flaky (≈ 1 in 5 runs failed). Cause: busy arrives in two shapes.
  Inside MCP the tool returns `assist is busy`; at the HTTP layer the middleware answered 503 with
  `{"error": "busy"}`, which the MCP client turns into a bare "Server returned an error response"
  (it surfaces a non-2xx body only when that body is a JSON-RPC error). Fix: the middleware's 503
  body is now a JSON-RPC error (`code -32000`, message `assist is busy (store queue full); retry
  in a few seconds`, `data.retry_after_s`), so an agent reads the retry hint either way, and the
  test client treats both shapes as busy. After the fix: 12/12 green.

## Alternatives not taken

- Raise `service.max_waiting` to ≥ 20: contradicts §6.3 (16) and only moves the edge.
- Drop the 503 bullet: loses the check that backpressure works.

## Reversibility

Test-only plus one response body. If the owner wants "zero first-try refusals at 20", set
`service.max_waiting: 20` in the deployment config and amend §6.3. Nothing else depends on it.
