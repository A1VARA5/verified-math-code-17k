# Pipeline

Five stages. Every script is in `pipeline/`, and every one of them is byte identical to the copy
published under `build/` with the dataset on Hugging Face. `verify.py` check 16 proves that by
SHA-256, so nothing here is a cleaned-up retelling of what actually ran.

```
        seven public corpora in data/
                     |
        1. build_math.py   build_code.py   build_general.py
           verified by     verified by     not verified,
           gold answer     execution       on purpose
                     |
              2. harden.py                    quality gate, drop-reason tally
                     |
              3. expand_pools.py              more rows at the same bar
                     |
              4. assemble_v3.py               17,000 rows + build_manifest.json
                     |
              5. verify_published.py          re-check of the PUBLISHED rows
```

## Paths

`pipeline/common.py` resolves everything against the directory above itself, so run the scripts
from the repository root and they will read and write:

| Path | What goes there |
|---|---|
| `data/` | the raw corpus dumps, one JSONL per upstream dataset, plus the decontamination targets |
| `pools/` | stage 1 output |
| `pools_hardened/` | stage 2 output, set with `POOLS_DIR` |
| `pools_expanded/` | stage 3 output, only the new rows |
| `dataset/` | stage 4 output, set with `OUT_DIR` |

Requirements: Python 3.11 or newer, `sympy`, `datasketch`, `pyarrow`. Built and run on Python 3.14
on Windows.

## Getting the corpora

The build reads raw dumps from `data/` as JSONL, named `<owner>__<dataset>.jsonl`. Those files are
not in this repository because they are large and they belong to their publishers. Pull them
yourself:

```python
from datasets import load_dataset
import json

for repo, out in [
    ("nvidia/OpenMathReasoning",        "data/nvidia__OpenMathReasoning.jsonl"),
    ("nvidia/OpenCodeInstruct",         "data/nvidia__OpenCodeInstruct.jsonl"),
    ("nvidia/OpenMathInstruct-2",       "data/nvidia__OpenMathInstruct-2.jsonl"),
    ("allenai/tulu-3-sft-mixture",      "data/allenai__tulu-3-sft-mixture.jsonl"),
    ("open-r1/OpenR1-Math-220k",        "data/open-r1__OpenR1-Math-220k.jsonl"),
    ("databricks/databricks-dolly-15k", "data/databricks__databricks-dolly-15k.jsonl"),
    ("deepmind/code_contests",          "data/deepmind__code_contests.jsonl"),
]:
    ds = load_dataset(repo, split="train")
    with open(out, "w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps(r) + "\n")
```

The decontamination targets go in the same folder: `HELDOUT__gsm8k_test.jsonl` (GSM8K test, 1,319
rows), `HELDOUT__mbpp_test.jsonl` (MBPP test, `task_id` 11 to 510, 500 rows) and
`openai__openai_humaneval.jsonl` (164 rows). MBPP and HumanEval are decontamination targets only.
No row in the published dataset derives from either.

---

## 1. The pools

### `build_math.py`

**What it does.** Scans the public math corpora, extracts the final answer each worked solution
arrives at, and keeps the row only if that answer is equivalent to a gold answer. Where the gold
answer is independent of the solution being kept (OpenMathReasoning, OpenR1-Math,
OpenMathInstruct-2, GSM8K) the row is marked `verify_level: "independent"`, which is a real
correctness check. Sources that carry only their own boxed answer are marked `"self"`, and none of
those survived into the published set. Hard problems with long worked reasoning are preferred.

Equivalence is `verify_core.answers_equivalent`, three escalating tiers:

1. normalised string equality, which catches the vast majority
2. numeric equality over floats and fractions, small relative tolerance
3. symbolic equality via SymPy, `simplify(gold - pred) == 0`, guarded by a thread timeout

`math_verify` is deliberately not used. Its comparison backend silently returned `False` on this
Windows and Python 3.14 setup, verified empirically, so the equivalence logic here is owned rather
than delegated. The cost of owning it is documented: `verify_core.normalize` does not strip LaTeX
inline-math delimiters, so a gold answer stored as `\(\infty\)` will not match a boxed `\infty`.
Nine published rows are affected and all nine are flagged in the verification manifest.

**Out.** `pools/math.jsonl` with `problem`, `worked_solution`, `gold`, `source`, `difficulty`,
`verify_level`, `domain`.

### `build_code.py`

**What it does.** Verification here is execution, not inspection. Two routes:

- **OpenCodeInstruct**, assert-style unit tests. Each candidate solution is written to a temp file
  and run in a fresh subprocess with an assert wrapper. Every test must pass, `min_ratio=1.0`.
  A row failing one assert is dropped.
- **DeepMind code_contests**, competition problems with stdin and stdout public tests. Up to six
  reference solutions are tried per problem and the first that reproduces every public test
  exactly, on whitespace-normalised comparison, is kept.

Each candidate runs in its own short-lived subprocess with an 8 second wall-clock timeout, so an
infinite loop or a crash in third-party dataset code kills its own child and never the build. That
is containment, not a sandbox. Run this stage in a container or a throwaway VM.

**Out.** `pools/code.jsonl`, every row `verify_level: "executed"`.

### `build_general.py`

**What it does.** Builds the unverified general slice from Tulu-3 and Dolly-15k, as anti-forgetting
ballast. It deliberately excludes math and code flavoured rows, by regex on both prompt and
response, so the ballast preserves broad instruction-following without diluting or duplicating the
verified domain signal. Nothing here is verified and the shipped rows say so with
`verify_level: "-"`.

**Out.** `pools/general.jsonl`.

```bash
python pipeline/build_math.py
python pipeline/build_code.py
python pipeline/build_general.py
```

---

## 2. `harden.py`

**What it does.** A quality gate over the pools with a per-reason drop tally, reading `pools/` and
writing `pools_hardened/` so the originals stay intact. Filters, in order:

| Filter | Drops |
|---|---|
| `clean` | strips `<think>` tag residue and whitespace, keeps the reasoning content |
| `refusal` | assistant refusals, "as an AI language model" and friends |
| `cjk` | non-English problems by CJK codepoint |
| `degenerate` | repetition collapse, eight or more identical consecutive lines |
| `decontam` | any row whose problem shares a 13-gram with GSM8K test, MBPP test or HumanEval |
| `math_wrong` | math whose boxed answer is numerically unequal to gold |
| `code_broken` | no code block, or Python that will not compile |

```bash
python pipeline/harden.py
```

---

## 3. `expand_pools.py`

**What it does.** Adds new verified rows on top of `pools_hardened/` without touching them, to
clear the volume floor for the challenge. Same bar as stage 1, plus the two things the first build
got wrong, fixed here:

- **Decontamination now runs against the real held-out splits.** The original build pointed at the
  *train* splits of GSM8K and MBPP, so the actual benchmarks had never been checked. GSM8K-train
  rows are no longer exempt. Two contaminated rows were found in the candidate pool and removed.
  Both were GSM8K-train rows that genuinely overlap GSM8K-test, which is an artefact of GSM8K
  itself rather than of this pipeline.
- **The hardening filters run here** rather than in a later pass.

It never re-emits a prompt already present in the hardened pools.

**Out.** `pools_expanded/{math,code,general}.jsonl`, containing only the new rows.

```bash
python pipeline/expand_pools.py --domain math    --target 6000
python pipeline/expand_pools.py --domain code    --target 1500
python pipeline/expand_pools.py --domain general --target 1200
```

---

## 4. `assemble_v3.py`

**What it does.** Composes the final 17,000 rows from the hardened and expanded pools, deduplicates
on the normalised prompt, runs the held-out decontamination once more, enforces the output
contract, and writes `build_manifest.json`.

**On the output contract.** Every math row must reach a `\boxed{}` answer. Rows carrying a verified
gold but no boxed answer get the canonical final line appended. The gold is already answer-verified
at that point, so appending it is formatting rather than fabrication. Rows with neither a boxed
answer nor a verified gold are dropped. This holds on all 9,104 published math rows.

What does **not** hold is the stronger claim the manifest's `output_contract` field states, that
every math row ends with the exact sentence `The final answer is $\boxed{...}$.`. Measured on the
published parquet, 1,006 of 9,104 do. The other 8,098 reach a boxed answer by a different surface
form, most commonly a display-math close with the box on the line above. Grade on the last
`\boxed{}`, never on the sentence. The manifest ships with the overstatement intact because it is
the build's own output record and it is published byte identical.

Near-duplicate removal uses `datasketch.MinHashLSH(threshold=0.6, num_perm=64)` over word 5-grams
of the normalised prompt. That index is approximate: re-running it over the shipped file flags
nothing, while an exhaustive exact-Jaccard sweep over the same shingles finds 26 pairs across 49
rows at Jaccard 0.6 or above, top 0.765. They are template siblings, the same problem skeleton with
different numbers, not copies. Zero prompts are byte-identical and zero are identical after
normalisation.

**Out.** the 17,000-row dataset plus `build_manifest.json`, which ships here as
[`evidence/build_manifest.json`](evidence/build_manifest.json).

```bash
python pipeline/assemble_v3.py --target 17000
```

---

## 5. `verify_published.py`

**What it does.** Nothing in the build. This is the script a third party runs to check the headline
claim without trusting anything on the dataset card. It downloads the published parquet and the
published `extras/verification_manifest.jsonl`, then re-applies the original two checks to the rows
exactly as shipped:

- **math**, 9,104 rows: extract the final answer from the published `worked_solution` and compare
  it against the independent gold answer by exact, numeric and symbolic equivalence, through the
  same code path as `verify_core.py`.
- **code**, 4,156 rows: run the published solution in a fresh subprocess against the unit tests it
  came with upstream. Every test must pass, same code path as `verify_code.py`.
- **general**, 3,740 rows: not checked. They carry `verify_level: "-"`, they are counted and
  skipped, and the card says so.

Every manifest entry carries the SHA-256 of the problem string it belongs to, so a gold answer or a
test set cannot drift onto the wrong row without the script reporting a binding failure rather than
passing over it.

```bash
pip install pyarrow sympy
python pipeline/verify_published.py                 # everything
python pipeline/verify_published.py --domain math   # math only, about a minute
python pipeline/verify_published.py --sample 300    # quick spot check
```

Expected output, and the reason it is not 13,260 of 13,260:

```
manifest-to-row binding failures: 0
math verified 9,095 / 9,104          deterministic, same number every run
code verified 4,147 to 4,149 / 4,156 moves between runs
verified about 13,243 / 13,260 checkable rows
3,740 general rows carry verify_level '-' and are not checkable
```

The 21 rows behind that gap are listed in the README and derived into
[`evidence/recheck_flags.json`](evidence/recheck_flags.json). Nine are a LaTeX delimiter gap in the
normaliser, one is a code row that fails every time, and eleven are upstream unit tests that are
nondeterministic.

This stage executes untrusted third-party code, same warning as stage 1.

---

## What is deterministic and what is not

Stages 2, 3 and 4 are deterministic functions of their inputs: the filters are pure, the
decontaminator is a set membership test over cached n-grams, and the dedup key is a normalised
string. Stage 1 depends on what the upstream corpora served on the day and on the behaviour of the
Python that executed the code candidates, which is exactly why the shipped verification manifest
records a SHA-256 per problem string rather than asking anyone to trust that a corpus still returns
the same rows.

`verify.py` at the repository root depends on none of it. It is standard library only, so checking
the claims never requires trusting the dependency tree that produced them.
