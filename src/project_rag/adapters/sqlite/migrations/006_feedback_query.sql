-- 006: feedback keeps the raw query text (spec §6.9: rescue/F-candidate labels need it).
-- Stays on the host under the transcript retention (§8.6).
ALTER TABLE feedback ADD COLUMN query_text TEXT NOT NULL DEFAULT '';
