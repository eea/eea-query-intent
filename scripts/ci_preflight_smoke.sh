#!/bin/bash
# Preflight: run the integration smoke against the existing release image,
# exactly as the Jenkins Integration stage will (run, cp, exec, cp back,
# stop, rm -v).
set -e
cd "$(dirname "$0")/.."
IMG="eea-query-intent:1.0.0"
C="qi-smoke-preflight"
docker rm -f "$C" 2>/dev/null || true
docker run -d --name "$C" "$IMG"
set +e
docker cp scripts/ci_smoke.py "$C":/tmp/ci_smoke.py
docker exec "$C" python /tmp/ci_smoke.py
RC=$?
docker cp "$C":/app/junit-smoke.xml /tmp/qi-smoke-junit.xml
set -e
docker stop "$C"
docker rm -v "$C"
echo "=== smoke exit: $RC ==="
echo "=== junit: ==="
cat /tmp/qi-smoke-junit.xml 2>/dev/null || echo "(no junit file)"
exit $RC
