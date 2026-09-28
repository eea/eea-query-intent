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
  # Same strips the Jenkins Unit stage applies before sonar-scanner runs:
  # drop the absolute-CWD <sources> element, then strip the src/ prefix
  # from the class filenames (sonar.sources=./src resolves them from src/).
  # NOTE: this whole block sits inside the outer single-quoted sh -c
  # string, so only double quotes (escaped for the inner shell) are allowed.
  sed -i "/<sources>/,/<\\/sources>/d" coverage/cobertura-coverage.xml
  sed -i "s|filename=\"src/|filename=\"|g" coverage/cobertura-coverage.xml
  echo "=== cobertura <sources> element (must be gone):"
  grep -c "<sources>" coverage/cobertura-coverage.xml || echo "0 (stripped)"
  echo "=== leftover src/ prefixes (must be 0):"
  grep -c "filename=\"src/" coverage/cobertura-coverage.xml || echo "0 (stripped)"
  echo "=== junit testcase count:"
  grep -c "<testcase" junit.xml || true
  echo "=== lcov SF lines:"
  grep "^SF:" coverage/lcov.info | head -4
  echo "=== cobertura filenames:"
  grep -o "filename=\"[^\"]*\"" coverage/cobertura-coverage.xml | head -5
'
