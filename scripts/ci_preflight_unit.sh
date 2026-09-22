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
  echo "=== junit.xml (first 15 lines):"
  head -15 junit.xml
  echo "=== junit testcase count:"
  grep -c "<testcase" junit.xml || true
  echo "=== lcov SF lines:"
  grep "^SF:" coverage/lcov.info | head -4
  echo "=== cobertura filenames:"
  grep -o "filename=\"[^\"]*\"" coverage/cobertura-coverage.xml | head -5
'
