#!/bin/sh
# Drive -> /corpus sync (spec §8.5). Runs in the rclone image (busybox sh + wget).
#  1. rclone sync with the media/temp filters and --max-delete (a mass deletion in Drive stops
#     here instead of wiping the corpus; it is an alert, the operator decides).
#  2. Writes the sidecar /corpus/.sync/lsjson.json (rclone lsjson -R --original: Drive ids,
#     stable across renames) for the assist project registry.
#  3. If any file changed, POST /admin/ingest. Always push a Kuma heartbeat (down on failure).
set -u
REMOTE="${RCLONE_REMOTE:?RCLONE_REMOTE}"
DEST="${CORPUS_DIR:-/corpus}"
FILTERS="${FILTERS:-/deploy/rclone/filters.txt}"
MAX_DELETE="${MAX_DELETE:-50}"
LOG="$(mktemp)"

push() {  # $1 = up|down, $2 = message
    [ -n "${KUMA_PUSH_URL:-}" ] || return 0
    wget -q -O /dev/null "${KUMA_PUSH_URL}?status=$1&msg=$(echo "$2" | tr ' ' '+')" || true
}

rclone sync "$REMOTE" "$DEST" \
    --drive-export-formats docx,xlsx \
    --filter-from "$FILTERS" \
    --max-delete "$MAX_DELETE" \
    --fast-list -v 2>"$LOG"
rc=$?
changed=$(grep -cE ': (Copied|Updated|Deleted|Moved)' "$LOG" || true)
if [ "$rc" -ne 0 ]; then
    reason="rclone exit $rc"
    grep -q "max-delete" "$LOG" && reason="max-delete threshold ($MAX_DELETE) reached: refusing mass deletion"
    echo "SYNC FAILED: $reason"
    tail -5 "$LOG"
    push down "$reason"
    exit "$rc"
fi

mkdir -p "$DEST/.sync"
if rclone lsjson -R --original --fast-list "$REMOTE" \
        --filter-from "$FILTERS" > "$DEST/.sync/lsjson.json.tmp" 2>>"$LOG"; then
    mv "$DEST/.sync/lsjson.json.tmp" "$DEST/.sync/lsjson.json"
else
    echo "WARN: lsjson failed; registry falls back to folder-name keys"
fi

echo "SYNC OK: changed=$changed"
if [ "$changed" -gt 0 ] && [ -n "${ADMIN_URL:-}" ]; then
    if wget -q -O - --post-data='' "${ADMIN_URL}/admin/ingest"; then
        echo
        echo "INGEST TRIGGERED"
    else
        echo "WARN: ingest trigger failed (assist down?); the nightly reconcile will catch up"
    fi
fi
push up "changed=$changed"
