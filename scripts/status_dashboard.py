#!/usr/bin/env python3
"""Temporary real-time status dashboard for the eea-query-intent pipeline.

Read-only. Serves a single auto-refreshing page on 127.0.0.1:8787.

Run:  uv run python scripts/status_dashboard.py
Stop: pkill -f status_dashboard.py
"""

import contextlib
import json
import re
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8787
TRAIN_TARGET = 3000
EXAM_TARGET = 760

NOISE_RE = re.compile(
    r"nspkg|AttributeError|loader|Remainder|NoneType|Traceback|File \""
    r"|Error processing|Remainder of file"
)


def tail_lines(path: str | Path, n: int = 3, max_bytes: int = 1 << 16) -> list[str]:
    try:
        p = Path(path)
        if not p.exists():
            return []
        data = p.read_bytes()[-max_bytes:].decode("utf-8", "replace")
        out = []
        for line in data.splitlines():
            line = line.strip()
            if line and not NOISE_RE.search(line):
                out.append(line)
        return out[-n:]
    except OSError:
        return []


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def running_procs() -> dict[str, list[str]]:
    """Map 'pattern' -> list of 'lang-or-cmd' for each pipeline worker family."""
    found: dict[str, list[str]] = {}
    for pattern in (
        "gpt_train_gen.py",
        "gpt_train_qa.py",
        "gpt_acceptance_qa.py",
        "run_final_build.sh",
        "run_train_phase.sh",
        "bash scripts/train_one_lang.sh",
        "run_acceptance_qa_phase.sh",
        "eea_query_intent.service",
    ):
        res = subprocess.run(["pgrep", "-fl", pattern], capture_output=True, text=True)
        langs = []
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                langs.append(parts[-1])
        found[pattern] = langs
    return found


def training_state(lang: str, procs: dict[str, list[str]]) -> dict:
    d = ROOT / "data" / "training" / "v1"
    final = d / f"{lang}.jsonl"
    raw = d / f"{lang}.raw.jsonl"
    prog = d / f"{lang}.qa_progress"
    if final.exists() and count_lines(final) >= 0.95 * TRAIN_TARGET:
        return {"phase": "complete", "rows": count_lines(final)}
    running_qa = lang in procs.get("gpt_train_qa.py", [])
    running_gen = lang in procs.get("gpt_train_gen.py", [])
    if running_qa or prog.exists():
        return {
            "phase": "reviewing",
            "batches": count_lines(prog) if prog.exists() else 0,
            "batches_total": 15,
            "running": running_qa,
            "log": tail_lines(f"/tmp/trn_1_{lang}.log", 2),
        }
    if raw.exists():
        log = tail_lines(f"/tmp/trn_1_{lang}.log", 3)
        current = None
        for line in reversed(log):
            match = re.match(rf"{lang}/(\w+) (\d+)/(\d+)", line)
            if match:
                current = list(match.groups())
                break
        return {
            "phase": "generating",
            "rows": count_lines(raw),
            "rows_target": TRAIN_TARGET,
            "current": current,
            "running": running_gen,
            "log": log,
        }
    return {
        "phase": "queued",
        "running": running_gen,
        "log": tail_lines(f"/tmp/trn_1_{lang}.log", 1),
    }


def exam_state(lang: str, procs: dict[str, list[str]]) -> dict:
    d = ROOT / "data" / "acceptance" / "v1"
    final = d / f"{lang}.jsonl"
    if final.exists():
        return {"phase": "complete", "rows": count_lines(final)}
    prog = d / f"{lang}.qa_progress"
    raw = d / f"{lang}.raw.jsonl"
    running = lang in procs.get("gpt_acceptance_qa.py", [])
    if running or prog.exists():
        return {
            "phase": "reviewing",
            "batches": count_lines(prog) if prog.exists() else 0,
            "batches_total": 5,
            "running": running,
        }
    if raw.exists():
        return {"phase": "generated", "rows": count_lines(raw), "running": running}
    return {"phase": "pending", "running": running}


def final_build_state() -> dict:
    flag = ROOT / ".pipeline" / "final_build_done.flag"
    if flag.exists():
        out: dict = {"state": "complete"}
        thr = ROOT / "reports" / "final_threshold.json"
        if thr.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                out["threshold"] = json.loads(thr.read_text())
        out["summary"] = tail_lines(
            ROOT / "reports" / "final_build_summary.txt", 60, 4 << 16
        )
        return out
    procs = running_procs()
    if procs.get("run_final_build.sh"):
        return {"state": "running", "log": tail_lines("/tmp/final_build.log", 10)}
    return {"state": "waiting", "log": tail_lines("/tmp/final_build.log", 5)}


def service_health() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {"status": "unavailable"}


def build_status() -> dict:
    procs = running_procs()
    try:
        from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES

        langs = sorted(SUPPORTED_LANGUAGE_CODES)
    except Exception:
        langs = sorted(
            p.stem for p in (ROOT / "data" / "translations").glob("*.jsonl")
        ) + ["en"]
        langs = sorted(set(langs))

    routing: dict = {}
    rfile = ROOT / "data" / "train_routing.json"
    if rfile.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            routing = json.loads(rfile.read_text())

    training = {}
    gpt_paused = (ROOT / ".pipeline" / "gpt_paused").exists()
    for lang in langs:
        st = training_state(lang, procs)
        model = (
            "gpt"
            if routing.get(lang, {}).get("gen_model", "")
            and "gpt" in routing[lang]["gen_model"]
            else "inhouse"
        )
        st["model"] = model
        # While GPT is paused (user needs the Codex quota), GPT-routed
        # languages are checkpointed and idle: show that explicitly.
        if gpt_paused and model == "gpt" and st["phase"] != "complete":
            st["phase"] = "paused"
        training[lang] = st

    exam = {lang: exam_state(lang, procs) for lang in langs}
    relaunch = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True
    ).stdout
    agents = [ln for ln in relaunch.splitlines() if "eeaki" in ln]

    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "langs": langs,
        "training": training,
        "exam": exam,
        "final_build": final_build_state(),
        "service": service_health(),
        "watchdog": {
            "last_check": tail_lines("/tmp/eeaki-relaunch-stdout.log", 1),
            "agents": agents,
        },
        "train_phase_running": bool(
            procs.get("run_train_phase.sh")
            or procs.get("bash scripts/train_one_lang.sh")
        ),
        "gpt_paused": gpt_paused,
    }


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>eea-query-intent pipeline</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --purple: #bc8cff; --red: #f85149;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 20px; background: var(--bg); color: var(--text);
         font: 14px/1.45 -apple-system, "Segoe UI", sans-serif; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
       color: var(--muted); margin: 24px 0 8px; }
  .topbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
  .pill { background: var(--panel); border: 1px solid var(--border);
          border-radius: 20px; padding: 5px 14px; font-size: 13px; }
  .pill b { color: var(--green); }
  .grid { display: grid; grid-template-columns: 1fr 340px; gap: 20px; }
  @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; background: var(--panel);
          border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
       color: var(--muted); background: #1c2129; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; border-radius: 4px; padding: 1px 8px;
           font-size: 11px; font-weight: 600; }
  .badge.gpt { background: #3d2a56; color: var(--purple); }
  .badge.inhouse { background: #12331c; color: var(--green); }
  .badge.paused { background: #4a3a12; color: #e0b34c; }
  .bar { background: #21262d; border-radius: 4px; height: 8px; width: 140px;
         overflow: hidden; display: inline-block; vertical-align: middle; }
  .bar i { display: block; height: 100%; border-radius: 4px; }
  .bar i.gen { background: var(--blue); }
  .bar i.qa { background: var(--yellow); }
  .bar i.done { background: var(--green); }
  .pct { font-size: 12px; color: var(--muted); margin-left: 6px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #30363d; margin-right: 6px; }
  .dot.live { background: var(--green); animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: .35; } }
  .log { color: var(--muted); font-size: 11px; font-family: ui-monospace, monospace;
         max-width: 420px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .phase { font-size: 12px; font-weight: 600; }
  .phase.complete { color: var(--green); }
  .phase.generating { color: var(--blue); }
  .phase.reviewing { color: var(--yellow); }
  .phase.paused { color: #e0b34c; }
  .phase.queued, .phase.pending, .phase.generated { color: var(--muted); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { border: 1px solid var(--border); border-radius: 6px; padding: 4px 9px;
          font-size: 12px; background: var(--panel); }
  .chip.complete { border-color: var(--green); color: var(--green); }
  .chip.work { border-color: var(--yellow); color: var(--yellow); animation: pulse 1.5s infinite; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 12px 14px; }
  pre { margin: 0; font: 11px/1.5 ui-monospace, monospace; color: var(--muted);
        white-space: pre-wrap; word-break: break-word; }
  .footer { margin-top: 24px; color: var(--muted); font-size: 12px; }
  .ok { color: var(--green); } .bad { color: var(--red); } .warn { color: var(--yellow); }
</style>
</head>
<body>
<h1>eea-query-intent &mdash; training scale-up</h1>
<div class="topbar" id="topbar"></div>
<div class="grid">
  <div>
    <h2>Training corpus (3000 rows per language)</h2>
    <table>
      <thead><tr>
        <th></th><th>lang</th><th>model</th><th>phase</th><th>progress</th><th>last activity</th>
      </tr></thead>
      <tbody id="trainbody"></tbody>
    </table>
    <h2>Final build</h2>
    <div class="panel" id="finalbuild"></div>
  </div>
  <div>
    <h2>Acceptance exam (760 rows per language)</h2>
    <div class="chips" id="examchips"></div>
    <h2>Service (127.0.0.1:8100)</h2>
    <div class="panel" id="service"></div>
    <h2>Watchdog</h2>
    <div class="panel" id="watchdog"></div>
  </div>
</div>
<div class="footer" id="footer">loading&hellip;</div>
<script>
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function pct(n, d) { return d > 0 ? Math.min(100, Math.round(100 * n / d)) : 0; }
function bar(p, cls) { return `<span class="bar"><i class="${cls}" style="width:${p}%"></i></span><span class="pct">${p}%</span>`; }
function modelBadge(m) { return m === 'gpt' ? '<span class="badge gpt">GPT</span>' : '<span class="badge inhouse">in-house</span>'; }

async function refresh() {
  let s;
  try { s = await (await fetch('/api/status')).json(); }
  catch (e) { document.getElementById('footer').textContent = 'status unavailable: ' + e; return; }

  const tDone = s.langs.filter(l => s.training[l].phase === 'complete').length;
  const eDone = s.langs.filter(l => s.exam[l].phase === 'complete').length;
  const fb = s.final_build.state;
  const svc = s.service;
  document.getElementById('topbar').innerHTML = `
    <span class="pill">training <b>${tDone}</b>/${s.langs.length}</span>
    <span class="pill">exam <b>${eDone}</b>/${s.langs.length}</span>
    <span class="pill">GPT ${s.gpt_paused ? '<span class="bad">paused</span>' : '<span class="ok">active</span>'}</span>
    <span class="pill">final build <span class="${fb === 'complete' ? 'ok' : fb === 'running' ? 'warn' : 'bad'}">${fb}</span></span>
    <span class="pill">service ${svc.status === 'ok' ? `<span class="ok">${esc(svc.model_version)} @ ${svc.abstain_threshold}</span>` : '<span class="bad">down</span>'}</span>`;

  const rows = s.langs.map(l => {
    const t = s.training[l];
    let phase, prog, log;
    if (t.phase === 'complete') { phase = 'complete'; prog = bar(100, 'done'); log = `${t.rows} rows`; }
    else if (t.phase === 'paused') {
      phase = 'paused';
      prog = t.batches ? bar(pct(t.batches, t.batches_total), 'qa')
           : t.rows ? bar(pct(t.rows, t.rows_target), 'gen')
           : bar(0, 'gen');
      log = 'GPT paused \u2014 resumes from checkpoint when re-enabled';
    } else if (t.phase === 'reviewing') {
      phase = 'reviewing';
      prog = bar(pct(t.batches, t.batches_total), 'qa');
      log = (t.log && t.log[0]) || `batch ${t.batches}/${t.batches_total}`;
    } else if (t.phase === 'generating') {
      phase = 'generating';
      const c = t.current;
      prog = bar(pct(t.rows, t.rows_target), 'gen');
      log = c ? `${t.rows}/${t.rows_target} + ${c[0]} ${c[1]}/${c[2]}` : `${t.rows}/${t.rows_target}`;
      if (t.log && t.log[0]) log += `  \u00b7  ${esc(t.log[0])}`;
    } else { phase = t.phase; prog = bar(0, 'gen'); log = (t.log && t.log[0]) || 'queued'; }
    return `<tr>
      <td><span class="dot ${t.running ? 'live' : ''}"></span></td>
      <td><b>${l}</b></td>
      <td>${modelBadge(t.model)}</td>
      <td class="phase ${phase}">${phase}</td>
      <td>${prog}</td>
      <td class="log" title="${esc(log)}">${esc(log)}</td>
    </tr>`;
  }).join('');
  document.getElementById('trainbody').innerHTML = rows;

  document.getElementById('examchips').innerHTML = s.langs.map(l => {
    const e = s.exam[l];
    const cls = e.phase === 'complete' ? 'complete' : (e.running || e.phase === 'reviewing' ? 'work' : '');
    const label = e.phase === 'complete' ? `${l} &#10003;` : e.phase === 'reviewing' ? `${l} ${e.batches}/${e.batches_total}` : e.phase === 'generated' ? `${l} raw` : l;
    return `<span class="chip ${cls}">${label}</span>`;
  }).join('');

  const fbd = s.final_build;
  document.getElementById('finalbuild').innerHTML =
    `<div class="phase ${fbd.state}">${fbd.state}</div>
     ${fbd.threshold ? `<div style="margin:6px 0">chosen threshold: <b>${esc(fbd.threshold.threshold)}</b> (fallback: ${fbd.threshold.fallback})</div>` : ''}
     <pre>${esc((fbd.summary || fbd.log || ['waiting for all 28 training finals + 28 exam shards']).join('\\n'))}</pre>`;

  const svcP = s.service;
  document.getElementById('service').innerHTML = svcP.status === 'ok'
    ? `<pre>${esc(JSON.stringify(svcP, null, 2))}</pre>`
    : `<span class="bad">unavailable</span>`;

  const w = s.watchdog;
  document.getElementById('watchdog').innerHTML =
    `<div>last check: ${esc((w.last_check[0] || 'unknown').replace('relaunch checked ', ''))}</div>
     <pre>${esc(w.agents.join('\\n') || 'no agents')}</pre>`;

  document.getElementById('footer').textContent =
    `updated ${s.now} &middot; refreshes every 5s &middot; read-only`;
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # silence
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self._send(json.dumps(build_status()).encode(), "application/json")
        elif path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"dashboard on http://127.0.0.1:{PORT} (ctrl-c to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
