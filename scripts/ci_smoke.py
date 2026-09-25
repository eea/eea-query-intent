"""Integration smoke test for the eea-query-intent release image.

Runs INSIDE the container (the Jenkins Integration stage copies it in with
`docker cp` and executes it with `docker exec python /tmp/ci_smoke.py`), so
it probes the real service over localhost: health, model identity, one
eligible query, one keyword query, and the deterministic policy guards.
Stdlib only — the release image has no test dependencies.

Writes a minimal JUnit XML to /tmp/junit-smoke.xml for Jenkins publishing.
Exit code 0 = all passed, 1 = at least one failure.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "http://127.0.0.1:8100"
# Written into the container workdir (root-owned, not world-writable like
# /tmp): Jenkins docker-cps the file back out of /app/junit-smoke.xml.
JUNIT_PATH = Path("/app/junit-smoke.xml")

ELIGIBLE_PROBE = "why are co2 emissions increasing in europe?"
KEYWORD_PROBE = "plastic waste"
TOO_LONG_QUERY = " ".join(
    [
        "what",
        "are",
        "the",
        "current",
        "co2",
        "emission",
        "figures",
        "for",
        "each",
        "member",
        "state",
        "under",
        "the",
        "emission",
        "trading",
        "scheme",
        "including",
        "corrections",
        "and",
        "adjustments",
        "please",
    ]
)
URL_QUERY = "https://example.com/page"


def post_classify(query: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{BASE}/v1/classify",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode())
        except json.JSONDecodeError:
            body = {"raw": str(error)}
        return error.code, body


def get(path: str, timeout: int = 30) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode())
        except json.JSONDecodeError:
            body = {"raw": str(error)}
        return error.code, body


class CollectingResult(unittest.TestResult):
    """TestResult that records an outcome for EVERY test, so the JUnit XML
    lists one <testcase> per test (the Jenkins junit step shows the full run
    instead of only the failures)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[unittest.TestCase, str, str]] = []

    def _record(self, test: unittest.TestCase, kind: str, detail: str = "") -> None:
        self.records.append((test, kind, detail))

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test: unittest.TestCase, err: tuple) -> None:
        super().addFailure(test, err)
        detail = (str(err[1]).strip() or err[0].__name__)[:500]
        self._record(test, "failed", detail)

    def addError(self, test: unittest.TestCase, err: tuple) -> None:
        super().addError(test, err)
        detail = (str(err[1]).strip() or err[0].__name__)[:500]
        self._record(test, "error", detail)

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)


class SmokeTests(unittest.TestCase):
    def test_health_reports_the_shipped_model(self) -> None:
        status, body = get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_version"])
        self.assertIn("abstain_threshold", body)

    def test_genuine_question_is_eligible(self) -> None:
        status, body = post_classify(ELIGIBLE_PROBE)
        self.assertEqual(status, 200)
        self.assertTrue(body["eligible"], body)
        self.assertIn(body["intent"], {"question", "exploratory", "claim"})

    def test_keyword_search_is_not_eligible(self) -> None:
        status, body = post_classify(KEYWORD_PROBE)
        self.assertEqual(status, 200)
        self.assertFalse(body["eligible"], body)

    def test_empty_query_fails_closed(self) -> None:
        status, body = post_classify("")
        self.assertEqual(status, 200)
        self.assertFalse(body["eligible"])
        self.assertEqual(body["reason"], "empty")

    def test_too_long_query_fails_closed(self) -> None:
        status, body = post_classify(TOO_LONG_QUERY)
        self.assertEqual(status, 200)
        self.assertFalse(body["eligible"])
        self.assertEqual(body["reason"], "too_long")

    def test_url_query_fails_closed(self) -> None:
        status, body = post_classify(URL_QUERY)
        self.assertEqual(status, 200)
        self.assertFalse(body["eligible"])
        self.assertEqual(body["reason"], "url")

    def test_wrong_method_and_path_are_rejected(self) -> None:
        status, _ = get("/v1/classify")
        self.assertEqual(status, 405)
        status, _ = get("/definitely/not/a/route")
        self.assertEqual(status, 404)


def write_junit(records: list[tuple[str, str, str]]) -> None:
    """Write one <testcase> per test (records: name, kind, detail)."""
    failures = sum(1 for _, kind, _ in records if kind == "failed")
    errors = sum(1 for _, kind, _ in records if kind == "error")
    skipped = sum(1 for _, kind, _ in records if kind == "skipped")
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "integration-smoke",
            "tests": str(len(records)),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
            "time": "0",
        },
    )
    for name, kind, detail in records:
        testcase = ET.SubElement(
            testsuite, "testcase", {"name": name, "classname": "integration-smoke"}
        )
        if kind == "failed":
            ET.SubElement(
                testcase, "failure", {"message": detail or "assertion failed"}
            )
        elif kind == "error":
            ET.SubElement(testcase, "error", {"message": detail or "unexpected error"})
        elif kind == "skipped":
            ET.SubElement(testcase, "skipped", {"message": detail})
    JUNIT_PATH.write_text(ET.tostring(testsuite, encoding="unicode"))
    print(f"junit written to {JUNIT_PATH}")


def main() -> int:
    # Wait for the service: the model cold-starts on first boot. Bounded by
    # a real wall-clock deadline with short per-attempt timeouts, so a hung
    # endpoint cannot stretch the stage for tens of minutes.
    deadline = time.monotonic() + 240
    healthy = False
    while time.monotonic() < deadline:
        try:
            status, _ = get("/health", timeout=3)
            if status == 200:
                healthy = True
                break
        except (OSError, ValueError):
            pass
        time.sleep(2)
    if not healthy:
        write_junit(
            [("service_start", "error", "service did not become healthy within 240s")]
        )
        return 1

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SmokeTests)
    result = CollectingResult()
    suite(result)

    records = [(str(test), kind, detail) for test, kind, detail in result.records]
    write_junit(records)
    ok = result.wasSuccessful() and all(kind == "passed" for _, kind, _ in records)
    for name, kind, detail in records:
        if kind != "passed":
            print(f"{kind}: {name}: {detail[:200]}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
