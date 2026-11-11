-- 005: feedback signals (spec §6.9; recorded from S6). No raw query text, only its hash.
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    project_id TEXT NOT NULL,
    at TEXT NOT NULL,
    query_hash TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_feedback_at ON feedback(at);
