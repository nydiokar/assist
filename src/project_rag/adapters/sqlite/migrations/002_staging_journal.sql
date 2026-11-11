-- 002: staging journal for the two-phase generation publish (S2, D-3 = Chroma).
-- A row is committed BEFORE a generation's rows/vectors are staged and removed in the same
-- transaction that flips the active pointer. Rows left behind mark a crashed staging run; the
-- startup GC deletes that generation from every index and then the row.
CREATE TABLE IF NOT EXISTS staged_generations (
    project_id TEXT NOT NULL,
    generation TEXT NOT NULL,
    started_at TEXT NOT NULL,
    PRIMARY KEY (project_id, generation)
);
