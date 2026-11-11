-- 004: ingest run reports (spec §6.5; written from S4). report_json holds counts by state, newly
-- failed documents with reasons, coverage %, per-stage durations and generation ids. Kept 90 days.
CREATE TABLE IF NOT EXISTS run_reports (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    prior_generation TEXT,
    new_generation TEXT,
    published INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_run_reports_project ON run_reports(project_id, started_at);
