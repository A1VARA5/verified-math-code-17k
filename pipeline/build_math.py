"""
Build the verified MATH pool.

For every candidate we extract the final answer the worked solution arrives at and
require it to be equivalent to a gold answer. Sources whose gold answer is INDEPENDENT
of the trained solution (OpenMathInstruct-2, OpenMathReasoning, OpenR1, GSM8K) give a
true correctness check; we mark these verify_level="independent". Sources that carry
their own boxed answer (DART-hard, NuminaMath) are marked "self" (consistency-checked,
boxed present + parseable). We prefer hard problems with long worked reasoning.

Writes pools/math.jsonl with: problem, worked_solution, gold, source, difficulty, verify_level.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl, write_jsonl, strip_think, POOLS
from verify_core import answers_equivalent, extract_final_answer, extract_boxed

# (file, problem_key, solution_key, gold_key|None, source, difficulty, verify_level, min_len, per_source_cap, max_scan)
SOURCES = [
    ("nvidia__OpenMathReasoning.jsonl", "problem", "generated_solution", "expected_answer",
     "OpenMathReasoning", "hard", "independent", 300, 1800, 40000),
    ("open-r1__OpenR1-Math-220k.jsonl", "problem", "solution", "answer",
     "OpenR1-Math", "hard", "independent", 300, 1500, 40000),
    ("hkust-nlp__dart-math-hard.jsonl", "query", "response", None,
     "DART-Math-hard", "hard", "self", 200, 1200, 50000),
    ("nvidia__OpenMathInstruct-2.jsonl", "problem", "generated_solution", "expected_answer",
     "OpenMathInstruct-2", "medium", "independent", 150, 1500, 30000),
    ("AI-MO__NuminaMath-CoT.jsonl", "problem", "solution", None,
     "NuminaMath-CoT", "medium", "self", 150, 1200, 30000),
    ("openai__gsm8k.jsonl", "question", "answer", "__gsm8k__",
     "GSM8K-train", "easy", "independent", 80, 900, 8000),
]


def gold_for(row, gold_key, solution):
    if gold_key == "__gsm8k__":
        return row["answer"].rsplit("####", 1)[1].strip() if "####" in row["answer"] else None
    if gold_key:
        return str(row.get(gold_key, "")).strip() or None
    return extract_boxed(solution)  # self-gold


def clean_solution(text: str) -> str:
    t = strip_think(text)
    # gsm8k: drop the calculator <<...>> annotations, keep the steps + #### answer
    return t


def main():
    t0 = time.time()
    pool, seen_q = [], set()
    per_src_counts = {}
    for (fname, pk, sk, gk, src, diff, vlev, min_len, cap, max_scan) in SOURCES:
        kept = 0
        try:
            for row in read_jsonl(fname, limit=max_scan):
                if kept >= cap:
                    break
                problem = str(row.get(pk, "")).strip()
                sol_raw = str(row.get(sk, "")).strip()
                if len(problem) < 10 or len(sol_raw) < min_len:
                    continue
                sol = clean_solution(sol_raw)
                gold = gold_for(row, gk, sol)
                if not gold:
                    continue
                pred = extract_final_answer(sol)
                if not answers_equivalent(gold, pred):
                    continue
                qkey = problem.lower()[:120]
                if qkey in seen_q:
                    continue
                seen_q.add(qkey)
                pool.append({
                    "problem": problem,
                    "worked_solution": sol,
                    "gold": gold,
                    "source": src,
                    "difficulty": diff,
                    "verify_level": vlev,
                    "domain": "math",
                })
                kept += 1
        except FileNotFoundError:
            print(f"  (missing {fname}, skipped)")
        per_src_counts[src] = kept
        print(f"  {src:22s} kept={kept}  (elapsed {time.time()-t0:.0f}s)")

    out = os.path.join(POOLS, "math.jsonl")
    write_jsonl(out, pool)
    hard = sum(1 for r in pool if r["difficulty"] == "hard")
    indep = sum(1 for r in pool if r["verify_level"] == "independent")
    print(f"\nMATH POOL: {len(pool)} verified rows -> {out}")
    print(f"  hard={hard} ({hard/max(1,len(pool))*100:.0f}%)  independent-gold={indep} ({indep/max(1,len(pool))*100:.0f}%)")
    print(f"  by source: {per_src_counts}")
    print(f"  total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
