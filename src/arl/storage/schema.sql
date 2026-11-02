-- Schema v1 de app.db (dominio + ledger de idempotencia + auditoría).
-- Las restricciones CHECK replican los modelos Pydantic: la base se defiende sola.

CREATE TABLE users (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
) STRICT;

CREATE TABLE projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE TABLE memberships (
    user_id    TEXT NOT NULL REFERENCES users (id),
    project_id TEXT NOT NULL REFERENCES projects (id),
    role       TEXT NOT NULL CHECK (role IN ('viewer', 'editor', 'owner')),
    PRIMARY KEY (user_id, project_id)
) STRICT;

CREATE TABLE tasks (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects (id),
    title       TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 200),
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL CHECK (status IN ('todo', 'in_progress', 'blocked', 'done')),
    assignee_id TEXT REFERENCES users (id),
    version     INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1)
) STRICT;

CREATE INDEX tasks_by_project ON tasks (project_id, id);

-- Una fila por escritura exitosa. Las claves están aisladas por principal.
CREATE TABLE idempotency_keys (
    principal_id  TEXT NOT NULL,
    key           TEXT NOT NULL,
    operation     TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (principal_id, key)
) STRICT;

CREATE TABLE audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    actor_id        TEXT NOT NULL,
    action          TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    before_json     TEXT NOT NULL,
    after_json      TEXT NOT NULL,
    UNIQUE (actor_id, idempotency_key)
) STRICT;

-- La auditoría es append-only.
CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log es append-only');
END;

CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log es append-only');
END;
