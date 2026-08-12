"""
verify_published.py — re-run this dataset's verification against the PUBLISHED rows.

This is the script a third party runs to check the card's headline claim without
trusting anything on the card. It does not rebuild the dataset. It downloads the
published parquet and the published verification manifest, then re-applies the two
original checks to the rows as shipped:

  math  (9,104 rows, verify_level "independent")
        extract the final answer out of the published worked_solution and compare it
        with the independent gold answer from the upstream corpus, by exact, numeric
        and symbolic (SymPy) equivalence. Same code path as build/verify_core.py.

  code  (4,156 rows, verify_level "executed")
        run the published solution in a fresh subprocess against the unit tests that
        shipped with it upstream. Every test must pass. Same code path as
        build/verify_code.py.

  general (3,740 rows, verify_level "-")
        not checked. These rows are unverified anti-forgetting ballast and the card
        says so. They are counted and skipped.

Every manifest entry carries the SHA-256 of the problem string it belongs to, so the
gold answer and the unit tests are bound to the exact published row. A mismatch is
reported as a binding failure rather than passed over.

Expected result on the published artifact:

    verified 13,260 / 13,260 checkable rows   (math 9,104, code 4,156)
    3,740 general rows carry verify_level "-" and are not checkable

Usage:

    pip install pyarrow sympy
    python build/verify_published.py                 # downloads both files, checks all
    python build/verify_published.py --domain math   # math only, about a minute
    python build/verify_published.py --sample 300    # quick spot check
    python build/verify_published.py --parquet local.parquet --manifest local.jsonl

Code verification runs real subprocesses. Budget roughly 10 to 25 minutes for the full
4,156 code rows on a laptop, or use --jobs to raise the worker count.
"""
from __future__ import annotations
import argparse, io, json, os, sys, time, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "manifesta/verified-math-code-17k"
PARQUET_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/train-00000-of-00001.parquet"
MANIFEST_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/extras/verification_manifest.jsonl"


def fetch(url: str, dest: str) -> str:
    if os.path.exists(dest):
        print(f"  using cached {dest}")
        return dest
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  wrote {dest} ({os.path.getsize(dest):,} bytes)")
    return dest


def load_rows(parquet_path: str) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path)
    return t.to_pylist()


# --------------------------------------------------------------------- code workers
def _run_stdio(code: str, inputs: list[str], outputs: list[str]) -> bool:
    """Competition rows: feed each public test on stdin, require matching stdout.
    Same rule as build/build_code.py, inlined here so this script stands alone."""
    import subprocess, tempfile
    from verify_code import NO_WINDOW, TIMEOUT
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
            if r.returncode != 0 or r.stdout.strip().split() != exp.strip().split():
                return False
    return True


def _check_code(args):
    idx, solution, kind, payload = args
    from verify_code import verify_code_row, extract_code
    if kind == "asserts":
        ok = verify_code_row(solution, payload, min_ratio=1.0)
    else:
        ok = _run_stdio(extract_code(solution), payload[0], payload[1])
    return idx, bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None, help="local parquet, else downloaded from the Hub")
    ap.add_argument("--manifest", default=None, help="local manifest, else downloaded from the Hub")
    ap.add_argument("--domain", default="all", choices=["all", "math", "code"])
    ap.add_argument("--sample", type=int, default=0, help="check only the first N rows per domain")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()

    print("=" * 74)
    print("VERIFIED MATH & CODE 17k — re-verification of the PUBLISHED rows")
    print("=" * 74)

    parquet = a.parquet or fetch(PARQUET_URL, "train-00000-of-00001.parquet")
    manifest = a.manifest or fetch(MANIFEST_URL, "verification_manifest.jsonl")

    rows = load_rows(parquet)
    print(f"\npublished rows: {len(rows):,}")
    entries = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    print(f"manifest entries: {len(entries):,}")

    unchecked = sum(1 for r in rows if r.get("verify_level") == "-")
    print(f"rows carrying verify_level '-' (unverified general ballast): {unchecked:,}")

    # ---- bind the manifest to the published rows by SHA-256 of the problem string
    binding_failures = []
    for e in entries:
        r = rows[e["row"]]
        sha = hashlib.sha256(r["problem"].encode("utf-8")).hexdigest()
        if sha != e["problem_sha256"] or r["verify_level"] != e["verify_level"]:
            binding_failures.append(e["row"])
    print(f"manifest-to-row binding failures: {len(binding_failures)}")

    math_e = [e for e in entries if e["domain"] == "math"]
    code_e = [e for e in entries if e["domain"] == "code"]
    if a.sample:
        math_e, code_e = math_e[: a.sample], code_e[: a.sample]

    math_ok = math_bad = code_ok = code_bad = 0
    bad_rows = []

    # ------------------------------------------------------------------ math
    if a.domain in ("all", "math"):
        from verify_core import answers_equivalent, extract_final_answer
        print(f"\n[math] re-checking {len(math_e):,} rows against independent gold answers")
        t0 = time.time()
        for i, e in enumerate(math_e, 1):
            sol = rows[e["row"]]["worked_solution"]
            pred = extract_final_answer(sol)
            if pred is not None and answers_equivalent(str(e["gold"]), pred):
                math_ok += 1
            else:
                math_bad += 1
                bad_rows.append(("math", e["row"]))
            if i % 1000 == 0:
                print(f"    {i:,}/{len(math_e):,}  ok={math_ok:,} bad={math_bad}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  math verified {math_ok:,} / {len(math_e):,}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ code
    if a.domain in ("all", "code"):
        print(f"\n[code] re-executing {len(code_e):,} published solutions against their unit tests")
        t0 = time.time()
        jobs = []
        for e in code_e:
            sol = rows[e["row"]]["worked_solution"]
            if e["test_kind"] == "asserts":
                jobs.append((e["row"], sol, "asserts", e["unit_tests"]))
            else:
                jobs.append((e["row"], sol, "stdio", (e["stdin"], e["stdout"])))
        done = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futs = [pool.submit(_check_code, j) for j in jobs]
            for fut in as_completed(futs):
                idx, ok = fut.result()
                done += 1
                if ok:
                    code_ok += 1
                else:
                    code_bad += 1
                    bad_rows.append(("code", idx))
                if done % 250 == 0:
                    print(f"    {done:,}/{len(jobs):,}  ok={code_ok:,} bad={code_bad}  "
                          f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  code verified {code_ok:,} / {len(code_e):,}  ({time.time()-t0:.0f}s)")

    total_ok = math_ok + code_ok
    total = len(math_e) * (a.domain in ("all", "math")) + len(code_e) * (a.domain in ("all", "code"))
    print("\n" + "=" * 74)
    print(f"verified {total_ok:,} / {total:,} checkable rows   "
          f"(math {math_ok:,}, code {code_ok:,})")
    print(f"{unchecked:,} general rows carry verify_level '-' and are not checkable")
    if bad_rows:
        print(f"FAILURES: {len(bad_rows)} — first 10 row indices: {bad_rows[:10]}")
    if binding_failures:
        print(f"BINDING FAILURES: {len(binding_failures)}")
    print("=" * 74)
    return 1 if (bad_rows or binding_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
