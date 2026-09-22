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
JUNIT_PATH = Path("/tmp/junit-smoke.xml")

ELIGIBLE_PROBE = "why are co2 emissions increasing in europe?"
KEYWORD_PROBE = "plastic waste"
TOO_LONG_QUERY = " ".join(
    ["what", "are", "the", "current", "co2", "emission", "figures", "for", "each", "member", "state", "under", "the", "emission", "trading", "scheme", "including", "corrections", "and", "adjustments", "please"]
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


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode())
        except json.JSONDecodeError:
            body = {"raw": str(error)}
        return error.code, body


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


def main() -> int:
    # Wait for the service: the model cold-starts on first boot.
    for _ in range(90):
        try:
            status, _ = get("/health")
            if status == 200:
                break
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    else:
        print("service did not become healthy within 180s", file=sys.stderr)
        return 1

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SmokeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Minimal JUnit XML for the Jenkins junit step.
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "integration-smoke",
            "tests": str(result.testsRun),
            "failures": str(len(result.failures)),
            "errors": str(len(result.errors)),
            "skipped": "0",
            "time": "0",
        },
    )
    for test, _ in result.failures + result.errors:
        testcase = ET.SubElement(testsuite, "testcase", {"name": str(test)})
        ET.SubElement(testcase, "failure" if test.id() in [str(t) for t, _ in result.failures] else "error")
    JUNIT_PATH.write_text(ET.tostring(testsuite, encoding="unicode"))
    print(f"junit written to {JUNIT_PATH}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
