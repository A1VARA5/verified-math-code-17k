"""
Code solution verification by EXECUTION.

A code row is kept only if its solution actually runs and passes the unit tests
shipped with it. We never trust a solution that merely looks right. Each candidate
is run in a fresh subprocess with a hard wall-clock timeout, so an infinite loop or
crash in dataset code can only kill its own short-lived child, never the build.

Covers the assert-style sources (OpenCodeInstruct: `unit_tests`; MBPP: `test_list`).
stdin/stdout competition sources (code_contests) are handled separately.
"""
from __future__ import annotations
import json, re, subprocess, sys, tempfile, os, textwrap

TIMEOUT = 8  # seconds per candidate
# Windows: run child python processes with no console window so the build does not
# flash hundreds of terminal windows on the user's screen.
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def extract_code(output: str) -> str:
    """Pull the python body out of a ```python ...``` fence, else return as-is."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", output, re.S)
    if m:
        return m.group(1).strip()
    return output.strip()


def _coerce_tests(tests) -> list[str]:
    if isinstance(tests, list):
        return [str(t) for t in tests]
    if isinstance(tests, str):
        s = tests.strip()
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            pass
        return [s]
    return []


def run_asserts(solution_code: str, tests, setup: str = "") -> tuple[int, int]:
    """Return (passed, total). Each test is an assert statement run after the solution."""
    tests = _coerce_tests(tests)
    if not tests:
        return (0, 0)
    harness = (
        setup + "\n" + solution_code + "\n\n"
        "import sys as _sys\n_p=0\n"
    )
    for t in tests:
        # each test isolated so one failure doesn't abort the rest
        harness += (
            "try:\n"
            + textwrap.indent(t.strip() or "pass", "    ")
            + "\n    _p+=1\nexcept Exception:\n    pass\n"
        )
    harness += "print('PASSED', _p)\n"

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "cand.py")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(harness)
        try:
            # NOT text=True: on Windows that decodes the child's stdout with the
            # locale codec (cp1252) and a single byte outside that map kills the
            # reader thread with UnicodeDecodeError, losing the row (and, before
            # this fix, aborting the whole build). Decode as UTF-8, replacing
            # anything undecodable — we only ever grep the output for "PASSED n".
            r = subprocess.run(
                [sys.executable, fp],
                capture_output=True, timeout=TIMEOUT, cwd=d,
                creationflags=NO_WINDOW,
            )
            stdout = (r.stdout or b"")[-65536:].decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            return (0, len(tests))
        except Exception:
            return (0, len(tests))
    m = re.search(r"PASSED (\d+)", stdout)
    passed = int(m.group(1)) if m else 0
    return (passed, len(tests))


def verify_code_row(output: str, tests, setup: str = "", min_ratio: float = 1.0) -> bool:
    """Keep the row only if ALL (min_ratio=1.0) of its tests pass on execution."""
    code = extract_code(output)
    if not code or len(code) < 5:
        return False
    passed, total = run_asserts(code, tests, setup)
    return total > 0 and (passed / total) >= min_ratio
