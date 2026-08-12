"""
Harden the pools: aggressive, defensible quality filtering applied to every row.
Reads the ORIGINAL pools/, writes hardened copies to pools_hardened/ (originals
untouched). Re-assemble with:  POOLS_DIR=pools_hardened python build/assemble_v2.py

Filters (per-row, with a drop-reason tally):
  * clean      : strip <think> tags + whitespace
  * refusal    : drop assistant refusals / "as an AI language model"
  * cjk        : drop non-English (CJK) problems
  * degenerate : drop repetition-collapsed solutions (>=8 identical consecutive lines)
  * decontam   : drop any row whose PROBLEM shares a 13-gram with GSM8K / MBPP / HumanEval
  * math_wrong : (math) drop rows whose boxed answer is numerically != gold
  * code_broken: (code) drop rows with no code block, or Python that won't compile
"""
from __future__ import annotations
import os, sys, re, json
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl, strip_think, ngrams, Decontaminator
from collections import Counter

HERE = os.path.dirname(__file__)
SRC_POOLS = os.path.join(HERE, "..", "pools")
DST_POOLS = os.path.join(HERE, "..", "pools_hardened")
os.makedirs(DST_POOLS, exist_ok=True)

CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
REFUSALS = [
    "as an ai language model", "i cannot assist", "i can't help with that",
    "i'm sorry, but i can", "i am unable to", "i cannot fulfill",
]

# ---------- filters ----------
def is_refusal(sol):
    low = sol.lower()
    return any(p in low for p in REFUSALS)

def is_degenerate(sol):
    lines = sol.split("\n")
    run = 1
    for a, b in zip(lines, lines[1:]):
        if a.strip() and a == b:
            run += 1
            if run >= 8:
                return True
        else:
            run = 1
    return False

# math answer re-check ------------------------------------------------
def last_boxed(sol):
    i = sol.rfind(r"\boxed")
    if i < 0:
        return None
    j = sol.find("{", i)
    if j < 0:
        return None
    d, k = 0, j
    while k < len(sol):
        if sol[k] == "{":
            d += 1
        elif sol[k] == "}":
            d -= 1
            if d == 0:
                return sol[j + 1:k]
        k += 1
    return None

_REPL = [(r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"), (r"\left", ""), (r"\right", ""),
         (r"\!", ""), (r"\,", ""), (r"\;", ""), (r"\(", ""), (r"\)", ""), (r"\[", ""),
         (r"\]", ""), (r"\text", ""), ("$", ""), (",", ""), (" ", ""), ("{", ""), ("}", "")]

def _norm(s):
    s = str(s)
    for a, b in _REPL:
        s = s.replace(a, b)
    return s.strip().lower()

def _num(s):
    try:
        from sympy import N
        from sympy.parsing.latex import parse_latex
        return float(N(parse_latex(str(s))))
    except Exception:
        pass
    try:
        return float(str(s).replace("%", "").replace("$", "").strip())
    except Exception:
        return None

def math_is_wrong(row):
    gold = row.get("gold")
    box = last_boxed(row["worked_solution"])
    if gold is None or box is None:
        return False  # can't disprove -> keep
    if _norm(gold) == _norm(box):
        return False
    a, b = _num(gold), _num(box)
    return a is not None and b is not None and abs(a - b) > 1e-6

# code compile check --------------------------------------------------
def code_broken(sol):
    blocks = re.findall(r"```(\w+)?\s*(.*?)```", sol, re.S)
    if not blocks:
        return True  # a code answer with no code block is broken
    lang, code = max(blocks, key=lambda b: len(b[1]))
    code = code.strip()
    l = (lang or "").lower()
    if l in ("cpp", "c++", "c", "java", "javascript", "js", "go", "rust", "ruby", "php"):
        return False  # non-python; trust build-time execution
    if "#include" in code or "int main(" in code:
        return False  # untagged C/C++
    try:
        compile(code, "<s>", "exec")
        return False
    except Exception:
        return True


def build_bench():
    d = Decontaminator(n=13)
    for r in read_jsonl("google-research-datasets__mbpp.jsonl"):
        d.add_benchmark_text(str(r.get("text", "")))
    for r in read_jsonl("openai__openai_humaneval.jsonl"):
        d.add_benchmark_text(str(r.get("prompt", "")))
    for r in read_jsonl("openai__gsm8k.jsonl"):
        d.add_benchmark_text(str(r.get("question", "")))
    return d


def harden_pool(name, bench):
    rows = list(read_jsonl(os.path.join(SRC_POOLS, f"{name}.jsonl")))
    kept, reasons = [], Counter()
    for r in rows:
        r = dict(r)
        r["worked_solution"] = strip_think(r.get("worked_solution", ""))
        prob = r.get("problem", "")
        sol = r["worked_solution"]

        if CJK.search(prob) or CJK.search(sol):
            reasons["cjk"] += 1; continue
        if is_refusal(sol):
            reasons["refusal"] += 1; continue
        if is_degenerate(sol):
            reasons["degenerate"] += 1; continue
        # GSM8K is a seed we intentionally keep; everything else decontaminated
        if r.get("source") != "GSM8K-train" and bench.is_contaminated(prob):
            reasons["decontam"] += 1; continue
        if name == "math" and math_is_wrong(r):
            reasons["math_wrong"] += 1; continue
        if name == "code" and code_broken(sol):
            reasons["code_broken"] += 1; continue
        kept.append(r)

    with open(os.path.join(DST_POOLS, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:8}: {len(rows)} -> {len(kept)}   dropped={dict(reasons)}")
    return len(rows), len(kept)


def main():
    bench = build_bench()
    tot_in = tot_out = 0
    for name in ("math", "code", "general"):
        a, b = harden_pool(name, bench)
        tot_in += a; tot_out += b
    print(f"\nTOTAL: {tot_in} -> {tot_out}  (dropped {tot_in - tot_out})")
    print(f"hardened pools written to {DST_POOLS}")
    print("re-assemble:  POOLS_DIR=pools_hardened python build/assemble_v2.py"
          "  &&  POOLS_DIR=pools_hardened python build/assemble_reserve.py")


if __name__ == "__main__":
    main()
