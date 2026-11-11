-- 003: project registry columns (spec §6.4; filled by ProjectRegistryService in S4).
-- drive_id = stable Drive folder id from the sync sidecar (dev fallback: hash of the folder path);
-- slug = project_id assigned once; display_name follows the folder name.
ALTER TABLE projects ADD COLUMN drive_id TEXT;
ALTER TABLE projects ADD COLUMN slug TEXT;
ALTER TABLE projects ADD COLUMN display_name TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_drive_id ON projects(drive_id);
