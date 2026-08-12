"""
Build the verified CODE pool by EXECUTION.

Backbone: OpenCodeInstruct (assert-style unit tests) — keep a row only if ALL its
tests pass when we actually run the solution. Hard slice: code_contests competition
problems — run the reference solution under python3 against the public tests (stdin ->
stdout) and keep it only if every public test matches.

MBPP and HumanEval are NOT used here — they are reserved as decontamination targets
(see assemble.py) so we can never be accused of training on a public benchmark.

Writes pools/code.jsonl with: problem, worked_solution, source, difficulty, verify_level="executed".
"""
from __future__ import annotations
import os, sys, time, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl, write_jsonl, POOLS
from verify_code import verify_code_row, extract_code, NO_WINDOW

TIMEOUT = 8


def run_stdio(code: str, inputs: list[str], outputs: list[str]) -> bool:
    if not inputs or len(inputs) != len(outputs):
        return False
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "s.py")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(code)
        for inp, exp in zip(inputs, outputs):
            try:
                r = subprocess.run([sys.executable, fp], input=inp, capture_output=True,
                                   text=True, timeout=TIMEOUT, cwd=d,
                                   creationflags=NO_WINDOW)
            except Exception:
                return False
            if r.returncode != 0:
                return False
            if r.stdout.strip().split() != exp.strip().split():
                return False
    return True


def build_opencodeinstruct(cap, max_scan):
    out, n, scanned = [], 0, 0
    for row in read_jsonl("nvidia__OpenCodeInstruct.jsonl", limit=max_scan):
        if n >= cap:
            break
        scanned += 1
        output = str(row.get("output", ""))
        tests = row.get("unit_tests", "")
        if verify_code_row(output, tests, min_ratio=1.0):
            prob = str(row.get("input", "")).strip()
            if len(prob) < 15:
                continue
            out.append({"problem": prob, "worked_solution": output.strip(),
                        "source": "OpenCodeInstruct", "difficulty": "medium",
                        "verify_level": "executed", "domain": "code"})
            n += 1
        if scanned % 500 == 0:
            print(f"  [OCI] scanned={scanned} kept={n} ({time.time():.0f})", flush=True)
    return out


def build_code_contests(cap):
    out, n = [], 0
    for row in read_jsonl("deepmind__code_contests.jsonl"):
        if n >= cap:
            break
        sols = row.get("solutions") or {}
        sol_list = sols.get("solution") or []
        pub = row.get("public_tests") or {}
        inputs, outputs = pub.get("input") or [], pub.get("output") or []
        desc = str(row.get("description", "")).strip()
        if not sol_list or not inputs or len(desc) < 20:
            continue
        for sol in sol_list[:6]:  # try a few reference solutions, keep first that passes
            if run_stdio(sol, inputs, outputs):
                ws = f"```python\n{sol.strip()}\n```"
                out.append({"problem": desc, "worked_solution": ws,
                            "source": "code_contests", "difficulty": "hard",
                            "verify_level": "executed", "domain": "code"})
                n += 1
                break
    return out


def main():
    t0 = time.time()
    print("building OpenCodeInstruct (execution-verified)...", flush=True)
    oci = build_opencodeinstruct(cap=2400, max_scan=12000)
    print(f"  OpenCodeInstruct kept={len(oci)} ({time.time()-t0:.0f}s)", flush=True)
    print("building code_contests (hard, stdio-verified)...", flush=True)
    cc = build_code_contests(cap=400)
    print(f"  code_contests kept={len(cc)} ({time.time()-t0:.0f}s)", flush=True)
    pool = oci + cc
    # dedup on prompt head
    seen, dedup = set(), []
    for r in pool:
        k = r["problem"].lower()[:120]
        if k in seen:
            continue
        seen.add(k); dedup.append(r)
    out = os.path.join(POOLS, "code.jsonl")
    write_jsonl(out, dedup)
    hard = sum(1 for r in dedup if r["difficulty"] == "hard")
    print(f"\nCODE POOL: {len(dedup)} executed-verified rows (hard={hard}) -> {out}")
    print(f"  total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
