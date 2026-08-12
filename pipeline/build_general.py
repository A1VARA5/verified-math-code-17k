"""
Build the GENERAL pool (anti-forgetting slice).

Reasoning-leaning general instruction-following from permissive sources (Tulu-3 SFT
mixture, Dolly-15k). We deliberately exclude math/code-flavoured rows here so the
general slice preserves broad capability without diluting (or duplicating) the
verified domain signal. No verification needed; these are conditioning, not the metric.

Writes pools/general.jsonl with: problem, worked_solution, source, domain="general".
"""
from __future__ import annotations
import os, sys, time, re
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl, write_jsonl, POOLS

CODEY = re.compile(r"```|def |class |import |println|console\.|#include|public static", re.I)
MATHY = re.compile(r"\\boxed|\\frac|\\sqrt|\$\$|integral|theorem", re.I)


def looks_domain(text: str) -> bool:
    return bool(CODEY.search(text) or MATHY.search(text))


def from_tulu(cap, max_scan):
    out, n = [], 0
    for row in read_jsonl("allenai__tulu-3-sft-mixture.jsonl", limit=max_scan):
        if n >= cap:
            break
        src = str(row.get("source", "")).lower()
        if any(k in src for k in ("math", "code", "gsm", "evol", "wizard")):
            continue
        msgs = row.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if not user or not asst:
            continue
        if not (15 <= len(user) <= 2000 and 120 <= len(asst) <= 4000):
            continue
        if looks_domain(user) or looks_domain(asst):
            continue
        out.append({"problem": user.strip(), "worked_solution": asst.strip(),
                    "source": "Tulu-3", "domain": "general"})
        n += 1
    return out


def from_dolly(cap):
    out, n = [], 0
    for row in read_jsonl("databricks__databricks-dolly-15k.jsonl"):
        if n >= cap:
            break
        instr = str(row.get("instruction", "")).strip()
        ctx = str(row.get("context", "")).strip()
        resp = str(row.get("response", "")).strip()
        if len(instr) < 15 or len(resp) < 60:
            continue
        prompt = instr + (("\n\n" + ctx) if ctx else "")
        if looks_domain(prompt) or looks_domain(resp):
            continue
        out.append({"problem": prompt, "worked_solution": resp,
                    "source": "Dolly-15k", "domain": "general"})
        n += 1
    return out


def main():
    t0 = time.time()
    pool = from_tulu(cap=2600, max_scan=50000) + from_dolly(cap=1200)
    # dedup on prompt head
    seen, dedup = set(), []
    for r in pool:
        k = r["problem"].lower()[:120]
        if k in seen:
            continue
        seen.add(k); dedup.append(r)
    out = os.path.join(POOLS, "general.jsonl")
    write_jsonl(out, dedup)
    print(f"GENERAL POOL: {len(dedup)} rows -> {out}  ({time.time()-t0:.0f}s)")
    from collections import Counter
    print("  by source:", dict(Counter(r["source"] for r in dedup)))


if __name__ == "__main__":
    main()
