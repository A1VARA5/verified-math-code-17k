"""
Math answer verification for the AutoScientist math/code dataset build.

We do NOT trust a solution because it looks plausible. A row is kept only if the
final answer the worked solution arrives at is provably equivalent to the dataset's
gold answer. Equivalence is checked in three escalating tiers:

  1. normalized string equality  (cheap, catches the vast majority)
  2. numeric equality            (floats / fractions, small relative tolerance)
  3. symbolic equality via sympy (simplify(gold - pred) == 0), thread-timeout guarded

math_verify is intentionally NOT used: its comparison backend silently returns False
on this Windows / Python 3.14 setup (verified empirically), so we own the logic.
"""
from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

import sympy
from sympy import simplify, nsimplify, Rational, sympify

_EXEC = ThreadPoolExecutor(max_workers=4)


# ----------------------------------------------------------------------------- extraction
def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def extract_final_answer(solution: str) -> str | None:
    """Best-effort final answer from a worked solution."""
    b = extract_boxed(solution)
    if b is not None:
        return b
    # gsm8k style: after ####
    if "####" in solution:
        return solution.rsplit("####", 1)[1].strip()
    # "the answer is X" / "= X" on the last non-empty line
    tail = [ln for ln in solution.strip().splitlines() if ln.strip()]
    if not tail:
        return None
    last = tail[-1]
    m = re.search(r"(?:answer is|equals|=)\s*\$?([^\$.]+?)\$?\s*\.?$", last, re.I)
    if m:
        return m.group(1).strip()
    m = re.findall(r"-?\d[\d,]*\.?\d*", last)
    return m[-1] if m else None


# ----------------------------------------------------------------------------- normalize
_REPL = [
    (r"\left", ""), (r"\right", ""), (r"\,", ""), (r"\!", ""), (r"\;", ""), (r"\ ", ""),
    (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"), (r"\cdot", "*"), (r"\times", "*"),
    (r"^\circ", ""), (r"\%", ""), (r"\$", ""), (r"$", ""),
]

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    for a, b in _REPL:
        s = s.replace(a, b)
    s = s.strip().rstrip(".")
    s = re.sub(r"\s+", "", s)
    # strip a single pair of wrapping braces / parens
    while len(s) >= 2 and s[0] in "({" and s[-1] in ")}":
        s = s[1:-1]
    # 1,234 -> 1234 (thousands separators, but not 1,2 coordinate pairs)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s


# ----------------------------------------------------------------------------- to-sympy
def _to_sympy_str(s: str) -> str:
    s = s.replace(r"\frac", "frac").strip()
    # frac{a}{b} -> ((a)/(b))
    def _frac(m):
        return f"(({m.group(1)})/({m.group(2)}))"
    for _ in range(6):  # nested fractions
        new = re.sub(r"frac\{([^{}]*)\}\{([^{}]*)\}", _frac, s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt", "sqrt", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**").replace("\\pi", "pi").replace("\\", "")
    return s


def _try_float(s: str):
    s = s.strip()
    try:
        return float(s)
    except Exception:
        pass
    m = re.fullmatch(r"\(?\s*(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)\s*\)?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            return None
    return None


def _sympy_equal(a: str, b: str) -> bool:
    ea, eb = sympify(_to_sympy_str(a)), sympify(_to_sympy_str(b))
    if ea == eb:
        return True
    return simplify(ea - eb) == 0


# ----------------------------------------------------------------------------- public api
# A timed-out future is NOT cancelled — the worker thread keeps grinding on the
# pathological simplify() forever. With a fixed 4-worker pool, four such expressions
# permanently occupy every worker, after which EVERY later comparison queues behind
# them and fails on timeout. Observed live: the expansion verified 2,600 rows in 195s
# on OpenMathReasoning, then jammed to a near-standstill on OpenR1's harder LaTeX.
# Two guards below: refuse to hand sympy the shapes that hang it, and rebuild the pool
# once it looks wedged.
_MAX_SYMPY_LEN = 60
_HANGS = re.compile(r"\\(?:begin|matrix|pmatrix|bmatrix|cases|int|sum|prod|lim|infty)|"
                    r"[{\[]\s*[^]}]*,[^]}]*[]}]|\.\.\.|\\ldots|\\cdots")
_consecutive_timeouts = 0


def _sympy_worth_trying(a: str, b: str) -> bool:
    if len(a) > _MAX_SYMPY_LEN or len(b) > _MAX_SYMPY_LEN:
        return False
    if _HANGS.search(a) or _HANGS.search(b):
        return False
    return True


def answers_equivalent(gold: str, pred: str, timeout: float = 3.0) -> bool:
    global _EXEC, _consecutive_timeouts
    if gold is None or pred is None:
        return False
    ng, npd = normalize(gold), normalize(pred)
    if ng == "" or npd == "":
        return False
    if ng == npd:
        return True
    fg, fp = _try_float(ng), _try_float(npd)
    if fg is not None and fp is not None:
        return abs(fg - fp) <= 1e-6 * max(1.0, abs(fg), abs(fp))
    if not _sympy_worth_trying(ng, npd):
        return False
    try:
        fut = _EXEC.submit(_sympy_equal, ng, npd)
        ok = bool(fut.result(timeout=timeout))
        _consecutive_timeouts = 0
        return ok
    except FTimeout:
        _consecutive_timeouts += 1
        if _consecutive_timeouts >= 8:
            # every worker is wedged on an un-killable simplify; abandon the pool
            # (its threads are daemonic and will die with the process) and start clean
            _EXEC = ThreadPoolExecutor(max_workers=4)
            _consecutive_timeouts = 0
        return False
    except Exception:
        return False


def verify_row(gold: str, solution: str) -> bool:
    return answers_equivalent(gold, extract_final_answer(solution))
