# assist developer/operator targets (spec A03 §8.2, §8.8). On Windows without make, run the
# recipe lines directly. Host deployment lives in deploy/ (compose.yaml, .env, deployment.yaml).

COMPOSE        := docker compose --env-file deploy/.env -f deploy/compose.yaml
COMPOSE_MODELS := docker compose -f deploy/dev/compose.models.yaml
TAG            ?= dev
BACKUP_DIR     ?= backups/$(shell date +%Y%m%d-%H%M%S)
RESTORE_INTO   ?= /tmp/assist-restore-check

.PHONY: models models-verify models-up models-down up down pull sync eval eval-gate \
        backup restore deploy rollback

## ---- models (S3) ----
models:            ## download every pinned GGUF and verify its sha256
	uv run python scripts/fetch_models.py
models-verify:     ## re-hash ./models, download nothing
	uv run python scripts/fetch_models.py --verify-only
models-up:         ## dev: llama-server pair on :8081/:8082
	$(COMPOSE_MODELS) up -d
models-down:
	$(COMPOSE_MODELS) down

## ---- host stack (S8) ----
up:                ## start the whole stack and wait until every container is healthy
	$(COMPOSE) up -d --wait
down:
	$(COMPOSE) down
pull:
	$(COMPOSE) pull
sync:              ## one Drive sync now (the systemd timer runs this every 15 min)
	$(COMPOSE) run --rm sync

## ---- quality (S7) ----
eval:
	uv run python scripts/eval_yp.py --config configs/base.yaml
eval-gate:          ## gold eval vs the promoted baseline (PROJECT, ROOT, QUESTIONS required)
	uv run python scripts/eval_gate.py --config configs/base.yaml --project "$(PROJECT)" --root "$(ROOT)" --questions "$(QUESTIONS)"

## ---- state (S8, R6) ----
backup:            ## snapshot the store + run reports out of the assist-state volume
	$(COMPOSE) exec -T assist python /app/scripts/state_backup.py backup --state /state --out /state/backup
	docker cp "$$($(COMPOSE) ps -q assist)":/state/backup "$(BACKUP_DIR)"
restore:           ## restore a backup into a scratch dir and integrity-check it
	uv run python scripts/state_backup.py restore --src "$(SRC)" --into "$(RESTORE_INTO)"

## ---- deploy / rollback (§8.8) ----
deploy:            ## make deploy TAG=x : backup, pull, up, wait healthy, record last good
	$(MAKE) backup
	ASSIST_IMAGE=ghcr.io/$${GHCR_OWNER:-nydiokar}/assist:$(TAG) $(COMPOSE) pull assist
	ASSIST_IMAGE=ghcr.io/$${GHCR_OWNER:-nydiokar}/assist:$(TAG) $(COMPOSE) up -d --wait
	echo "$(TAG)" > deploy/.last_good
rollback:          ## back to the tag in deploy/.last_good
	ASSIST_IMAGE=ghcr.io/$${GHCR_OWNER:-nydiokar}/assist:$$(cat deploy/.last_good) $(COMPOSE) up -d --wait
