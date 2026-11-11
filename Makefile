# assist developer/operator targets. S3 adds the model targets; S8 adds up/down/eval/backup/deploy.
# On Windows without make, run the recipe lines directly.

COMPOSE_MODELS := docker compose -f deploy/dev/compose.models.yaml

.PHONY: models models-verify models-up models-down

## Download every pinned GGUF (configs/models.lock.yaml) and verify its sha256.
models:
	uv run python scripts/fetch_models.py

## Re-hash what is in ./models; download nothing.
models-verify:
	uv run python scripts/fetch_models.py --verify-only

## Start / stop the local llama-server pair (embed :8081, rerank :8082).
models-up:
	$(COMPOSE_MODELS) up -d

models-down:
	$(COMPOSE_MODELS) down
