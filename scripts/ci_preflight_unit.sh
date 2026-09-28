#!/bin/bash
# Preflight: run the exact unit-test command inside the test image and
# verify the report artifacts the Jenkins pipeline will docker cp out.
set -e
cd "$(dirname "$0")/.."
docker run --rm eea-query-intent-test:preflight sh -c '
  uv run pytest --junitxml=junit.xml --cov=. \
    --cov-report=lcov:coverage/lcov.info \
    --cov-report=html:coverage/lcov-report \
    --cov-report=xml:coverage/cobertura-coverage.xml 2>&1 | tail -12
  # Same strip the Jenkins Unit stage applies before sonar-scanner runs.
  # The filenames keep their src/ prefix: file keys are root-relative
  # (src/eea_query_intent/...), so the report must match that shape.
  sed -i "/<sources>/,/<\\/sources>/d" coverage/cobertura-coverage.xml
  echo "=== cobertura <sources> element (must be gone):"
  grep -c "<sources>" coverage/cobertura-coverage.xml || echo "0 (stripped)"
  echo "=== cobertura filenames (must keep src/ prefix):"
  grep -o "filename=\"[^\"]*\"" coverage/cobertura-coverage.xml | head -3
  echo "=== junit testcase count:"
  grep -c "<testcase" junit.xml || true
  echo "=== lcov SF lines:"
  grep "^SF:" coverage/lcov.info | head -4
  echo "=== cobertura filenames:"
  grep -o "filename=\"[^\"]*\"" coverage/cobertura-coverage.xml | head -5
'
