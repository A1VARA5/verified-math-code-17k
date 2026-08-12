"""
Expand the verified pools to clear Sara Hooker's ~15k volume floor.

Adds NEW verified rows on top of pools_hardened/ without touching them. Everything a
row must survive to get in here is the same bar the original pools cleared, plus the
two things the original build got wrong:

  * decontamination now runs against the REAL held-out splits
    (HELDOUT__gsm8k_test.jsonl 1319, HELDOUT__mbpp_test.jsonl 500, HumanEval 164).
    The original build pointed at the TRAIN splits of GSM8K and MBPP, so the actual
    benchmarks were never checked. GSM8K-train rows are NO LONGER exempt.
  * hardening filters (CJK, refusals, <think>, degenerate repetition) are applied here
    rather than in a later pass.

Writes pools_expanded/{math,code,general}.jsonl = ONLY the new rows.

  python build/expand_pools.py --domain math    --target 6000
  python build/expand_pools.py --domain code    --target 1500
  python build/expand_pools.py --domain general --target 1200
"""
from __future__ import annotations
import os, sys, re, json, time, argparse
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl, write_jsonl, strip_think, normalize_prompt, ngrams, DATA
from verify_core import answers_equivalent, extract_final_answer, extract_boxed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HARD = os.path.join(ROOT, "pools_hardened")
OUTP = os.path.join(ROOT, "pools_expanded")
os.makedirs(OUTP, exist_ok=True)

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
REFUSAL = re.compile(
    r"\b(i (?:can'?t|cannot|am unable to) (?:help|assist|provide)|as an ai language model|"
    r"i'm sorry,? but i|i do not have (?:access|information))", re.I)


# --------------------------------------------------------------------------- guards
def degenerate(text: str) -> bool:
    """Same line repeated many times, or one token spammed.

    Counter, NOT max(set(xs), key=xs.count) — the latter is O(n^2) and these corpora
    contain 15KB reasoning traces, which made this single function the whole build's
    bottleneck (fast on short OpenMathInstruct rows, near-standstill on OpenR1).
    """
    lines = [l for l in (l.strip() for l in text.splitlines()) if l]
    if len(lines) >= 12:
        if Counter(lines).most_common(1)[0][1] / len(lines) > 0.45:
            return True
    toks = text.split()
    if len(toks) >= 60:
        if Counter(toks).most_common(1)[0][1] / len(toks) > 0.35:
            return True
    return False


def hygienic(problem: str, solution: str) -> bool:
    blob = problem + "\n" + solution
    if CJK.search(blob):
        return False
    if REFUSAL.search(solution[:400]):
        return False
    if "<think>" in blob or "</think>" in blob:
        return False
    if degenerate(solution):
        return False
    return True


class HeldOut:
    """13-gram decontaminator built from the ACTUAL benchmark test splits."""

    def __init__(self):
        self.g = set()
        n = {}
        for fname, fields in [
            ("HELDOUT__gsm8k_test.jsonl", ["question"]),
            ("HELDOUT__mbpp_test.jsonl", ["text", "prompt"]),
            ("openai__openai_humaneval.jsonl", ["prompt"]),
        ]:
            c = 0
            try:
                for row in read_jsonl(fname):
                    for fl in fields:
                        if row.get(fl):
                            self.g |= ngrams(str(row[fl]), 13)
                    c += 1
            except FileNotFoundError:
                print(f"  !! MISSING held-out file {fname} — decontam INCOMPLETE")
            n[fname] = c
        print(f"  held-out poison: {len(self.g):,} 13-grams from {n}")

    def hit(self, text: str) -> bool:
        g = ngrams(text, 13)
        return bool(g) and not g.isdisjoint(self.g)


def existing_keys() -> set:
    """Prompts already in the hardened pools — never re-emit one."""
    keys = set()
    for d in ("math", "code", "general"):
        p = os.path.join(HARD, f"{d}.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                keys.add(normalize_prompt(json.loads(line)["problem"])[:400])
    return keys


# --------------------------------------------------------------------------- math
# Ordered FAST-FIRST. OpenR1-Math-220k is deliberately last and scan-capped: it
# reliably grinds to a near-halt at the same point (~7.5k rows in) regardless of the
# degenerate()/sympy fixes, so it must never sit between us and the other five
# sources. Everything above it yields ~9k rows on its own.
MATH_SOURCES = [
    # file, problem_key, solution_key, gold_key, source, difficulty, verify_level, min_len, cap, max_scan
    ("nvidia__OpenMathReasoning.jsonl", "problem", "generated_solution", "expected_answer",
     "OpenMathReasoning", "hard", "independent", 300, 2600, 260000),
    ("nvidia__OpenMathInstruct-2.jsonl", "problem", "generated_solution", "expected_answer",
     "OpenMathInstruct-2", "medium", "independent", 150, 2200, 220000),
    ("microsoft__orca-math-word-problems-200k.jsonl", "question", "answer", None,
     "Orca-Math", "medium", "self", 150, 1800, 200000),
    ("meta-math__MetaMathQA.jsonl", "query", "response", None,
     "MetaMathQA", "medium", "self", 150, 1800, 200000),
    ("hkust-nlp__dart-math-hard.jsonl", "query", "response", None,
     "DART-Math-hard", "hard", "self", 200, 1400, 200000),
    ("open-r1__OpenR1-Math-220k.jsonl", "problem", "solution", "answer",
     "OpenR1-Math", "hard", "independent", 300, 800, 7000),
]


def gold_for(row, gold_key, solution):
    if gold_key:
        return str(row.get(gold_key, "")).strip() or None
    return extract_boxed(solution)


def expand_math(target, seen, bench):
    out, t0 = [], time.time()
    drops = {"dup": 0, "nogold": 0, "unverified": 0, "hygiene": 0, "decontam": 0, "short": 0}
    for (fname, pk, sk, gk, src, diff, vlev, min_len, cap, max_scan) in MATH_SOURCES:
        if len(out) >= target:
            break
        kept, scanned = 0, 0
        try:
            for row in read_jsonl(fname, limit=max_scan):
                if kept >= cap or len(out) >= target:
                    break
                scanned += 1
                problem = str(row.get(pk, "")).strip()
                sol_raw = str(row.get(sk, "")).strip()
                if len(problem) < 10 or len(sol_raw) < min_len:
                    drops["short"] += 1
                    continue
                key = normalize_prompt(problem)[:400]
                if key in seen:
                    drops["dup"] += 1
                    continue
                sol = strip_think(sol_raw)
                gold = gold_for(row, gk, sol)
                if not gold:
                    drops["nogold"] += 1
                    continue
                if not hygienic(problem, sol):
                    drops["hygiene"] += 1
                    continue
                if bench.hit(problem):
                    drops["decontam"] += 1
                    continue
                if not answers_equivalent(gold, extract_final_answer(sol)):
                    drops["unverified"] += 1
                    continue
                seen.add(key)
                out.append({"problem": problem, "worked_solution": sol, "gold": gold,
                            "source": src, "difficulty": diff, "verify_level": vlev,
                            "domain": "math"})
                kept += 1
                if kept % 400 == 0:
                    print(f"    [{src}] scanned={scanned} kept={kept} total={len(out)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        except FileNotFoundError:
            print(f"  (missing {fname}, skipped)")
        print(f"  {src:22s} +{kept:5d}  (scanned {scanned}, {time.time()-t0:.0f}s)", flush=True)
    print(f"  drops: {drops}")
    return out


# --------------------------------------------------------------------------- code
def expand_code(target, seen, bench):
    from verify_code import verify_code_row
    out, t0 = [], time.time()
    drops = {"dup": 0, "failed_tests": 0, "hygiene": 0, "decontam": 0, "short": 0}
    scanned, kept = 0, 0
    for row in read_jsonl("nvidia__OpenCodeInstruct.jsonl", limit=400000):
        if len(out) >= target:
            break
        scanned += 1
        prob = str(row.get("input", "")).strip()
        output = str(row.get("output", ""))
        if len(prob) < 15 or len(output) < 40:
            drops["short"] += 1
            continue
        key = normalize_prompt(prob)[:400]
        if key in seen:
            drops["dup"] += 1
            continue
        if not hygienic(prob, output):
            drops["hygiene"] += 1
            continue
        if bench.hit(prob):
            drops["decontam"] += 1
            continue
        if not verify_code_row(output, row.get("unit_tests", ""), min_ratio=1.0):
            drops["failed_tests"] += 1
            continue
        seen.add(key)
        out.append({"problem": prob, "worked_solution": output.strip(),
                    "source": "OpenCodeInstruct", "difficulty": "medium",
                    "verify_level": "executed", "domain": "code"})
        kept += 1
        if kept % 200 == 0:
            print(f"    [OCI] scanned={scanned} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  OpenCodeInstruct +{kept} (scanned {scanned}, {time.time()-t0:.0f}s)")
    print(f"  drops: {drops}")
    return out


# --------------------------------------------------------------------------- general
GENERAL_SOURCES = [
    ("allenai__tulu-3-sft-mixture.jsonl", "Tulu-3"),
    ("databricks__databricks-dolly-15k.jsonl", "Dolly-15k"),
]


def _general_pair(row):
    """Both corpora store either messages[] or instruction/response."""
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, list) and len(msgs) >= 2:
        u = next((m for m in msgs if m.get("role") in ("user", "human")), None)
        a = next((m for m in msgs if m.get("role") in ("assistant", "gpt")), None)
        if u and a:
            return str(u.get("content", "")).strip(), str(a.get("content", "")).strip()
    ins = str(row.get("instruction", "")).strip()
    ctx = str(row.get("context", "")).strip()
    res = str(row.get("response", "") or row.get("output", "")).strip()
    if ins and res:
        return (ins + ("\n\n" + ctx if ctx else "")).strip(), res
    return "", ""


def expand_general(target, seen, bench):
    out, t0 = [], time.time()
    drops = {"dup": 0, "hygiene": 0, "decontam": 0, "short": 0, "nonenglish": 0}
    NONLAT = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿ]")
    ES = re.compile(r"\b(el|la|los|las|una|que|para|como|con|por|del)\b", re.I)
    for fname, src in GENERAL_SOURCES:
        if len(out) >= target:
            break
        kept = 0
        try:
            for row in read_jsonl(fname, limit=200000):
                if len(out) >= target:
                    break
                prob, res = _general_pair(row)
                if len(prob) < 20 or len(res) < 80:
                    drops["short"] += 1
                    continue
                key = normalize_prompt(prob)[:400]
                if key in seen:
                    drops["dup"] += 1
                    continue
                # keep the general slice English-only this time round
                if NONLAT.search(prob[:300]) or len(ES.findall(prob[:300])) >= 4:
                    drops["nonenglish"] += 1
                    continue
                if not hygienic(prob, res):
                    drops["hygiene"] += 1
                    continue
                if bench.hit(prob):
                    drops["decontam"] += 1
                    continue
                seen.add(key)
                out.append({"problem": prob, "worked_solution": res,
                            "source": src, "domain": "general"})
                kept += 1
        except FileNotFoundError:
            print(f"  (missing {fname}, skipped)")
        print(f"  {src:22s} +{kept:5d}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  drops: {drops}")
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["math", "code", "general"])
    ap.add_argument("--target", type=int, required=True)
    a = ap.parse_args()

    print(f"=== EXPAND {a.domain.upper()} (target +{a.target}) ===", flush=True)
    bench = HeldOut()
    seen = existing_keys()
    print(f"  existing prompts to avoid: {len(seen):,}", flush=True)

    fn = {"math": expand_math, "code": expand_code, "general": expand_general}[a.domain]
    rows = fn(a.target, seen, bench)

    out = os.path.join(OUTP, f"{a.domain}.jsonl")
    write_jsonl(out, rows)
    print(f"\nEXPANDED {a.domain.upper()}: +{len(rows)} new verified rows -> {out}")


if __name__ == "__main__":
    main()
