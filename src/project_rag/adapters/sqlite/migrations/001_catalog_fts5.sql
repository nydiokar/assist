-- 001: the prototype schema (catalog + FTS5), unchanged. IF NOT EXISTS keeps pre-migration
-- databases (user_version 0) openable; their data is not migrated (L6 re-indexes from scratch).
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    active_generation TEXT,
    embedding_profile_key TEXT,
    collection TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    project_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version_id TEXT NOT NULL,
    state TEXT NOT NULL,
    generation TEXT NOT NULL,
    n_chunks INTEGER DEFAULT 0,
    diagnostics TEXT DEFAULT '',
    PRIMARY KEY (project_id, source_id, generation)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    generation TEXT NOT NULL,
    PRIMARY KEY (chunk_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_chunks_gen ON chunks(project_id, generation);
CREATE INDEX IF NOT EXISTS idx_docs_gen ON documents(project_id, generation);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    project_id UNINDEXED,
    generation UNINDEXED,
    document_id UNINDEXED,
    source_path UNINDEXED,
    heading UNINDEXED,
    location_json UNINDEXED,
    citation_text UNINDEXED,
    doc_class UNINDEXED,
    body,
    tokenize = 'unicode61 remove_diacritics 0'
);
