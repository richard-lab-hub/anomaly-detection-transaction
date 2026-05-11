#!/usr/bin/env bash
set -euo pipefail

# ── UID sync (root pass) ──────────────────────────────────────────────────────
# When started as root, remap appuser to match the host UID/GID of the bind
# mount, then drop privileges via gosu and re-exec this script as appuser.
if [ "$(id -u)" = "0" ]; then
    HOST_UID=$(stat -c %u /app 2>/dev/null || echo 1000)
    HOST_GID=$(stat -c %g /app 2>/dev/null || echo 1000)
    if [ "$HOST_UID" != "$(id -u appuser)" ] || [ "$HOST_GID" != "$(id -g appuser)" ]; then
        echo "[entrypoint] Remapping appuser → UID=$HOST_UID GID=$HOST_GID"
        groupmod -o -g "$HOST_GID" appuser >/dev/null 2>&1 || true
        usermod  -o -u "$HOST_UID" -g "$HOST_GID" appuser >/dev/null 2>&1 || true
        chown -R "$HOST_UID:$HOST_GID" /home/appuser
    fi
    exec gosu appuser "$0" "$@"
fi

# ── Below: running as appuser ────────────────────────────────────────────────

if [ -d /app/.git ] && command -v git-lfs >/dev/null 2>&1; then
    if git -C /app lfs ls-files 2>/dev/null | grep -q .; then
        echo "[entrypoint] git lfs pull (first-run data fetch)..."
        git -C /app lfs pull || echo "[entrypoint] WARNING: git lfs pull failed"
    fi
fi

if [ -f /app/pyproject.toml ] && [ ! -f /app/credit_risk.egg-info/PKG-INFO ]; then
    pip install -e /app --quiet
fi

if [ -n "${OLLAMA_HOST:-}" ]; then
    MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
    if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        if ! curl -sf "${OLLAMA_HOST}/api/tags" | grep -q "\"${MODEL}\""; then
            echo "[entrypoint] pulling Ollama model ${MODEL} (first run)..."
            curl -s "${OLLAMA_HOST}/api/pull" -d "{\"name\":\"${MODEL}\"}" \
                | grep -oE '"status":"[^"]+"' | tail -5 || true
            echo "[entrypoint] Ollama model ready."
        fi
    fi
fi

exec "$@"
