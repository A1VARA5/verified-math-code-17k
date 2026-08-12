# Verified Math and Code, 17,000 rows

[![verify](https://github.com/A1VARA5/verified-math-code-17k/actions/workflows/verify.yml/badge.svg)](https://github.com/A1VARA5/verified-math-code-17k/actions/workflows/verify.yml)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-verified--math--code--17k-2a78d6)](https://huggingface.co/datasets/manifesta/verified-math-code-17k)
[![weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-adaption__verified__math__code__instruct-2a78d6)](https://huggingface.co/manifesta/adaption_verified_math_code_instruct)
[![kaggle](https://img.shields.io/badge/Kaggle-mirror-20beff)](https://www.kaggle.com/datasets/aivarasnavardauskas/verified-math-code-17k)
[![licence](https://img.shields.io/badge/licence-CC--BY--SA--4.0-1baf7a)](LICENSE)

This is the build pipeline, the audit artifacts and the verification script behind
[`manifesta/verified-math-code-17k`](https://huggingface.co/datasets/manifesta/verified-math-code-17k),
a math and code instruction dataset built for the Adaption AutoScientist Challenge,
Math and Code category.

**13,260 of the 17,000 rows carry independent verification. The other 3,740 do not.** For the
math half I compared every worked solution's final answer against a gold answer that came from
outside that solution. For the code half I ran every solution against the unit tests it shipped
with upstream and required all of them to pass. The remaining 3,740 rows are general
instruction-following ballast, they carry `verify_level: "-"`, and nothing was ever checked about
them. That is 78.0% verified and 22.0% not, and both numbers are on this page because the second
one is not a rounding error.

## Check it yourself, do not take the README's word for it

```bash
git clone https://github.com/A1VARA5/verified-math-code-17k
cd verified-math-code-17k
python verify.py
```

No install step. `verify.py` is standard library only and runs on Python 3.9 or newer. It runs
23 checks against the audit files shipped here and against the live published artifacts, and
prints a pass or fail line for each one. Use `--offline` for the eight that need no network. The
same script runs in GitHub Actions on every push and once a day on a schedule, which is what the
badge at the top of this file reports.

What it checks:

| | Check |
|---|---|
| 1 | the build manifest ships here, parses, and is 825 bytes |
| 2 | four independent breakdowns in it each sum to 17,000: domain, verification level, difficulty, source |
| 3 | 13,260 verified plus 3,740 unverified equals 17,000, and 13,260 is 78.0% |
| 4 | 17,000 minus the 632 share-alike rows equals 16,368 |
| 5 | the recheck evidence adds up: 13,239 clean, 10 deterministic failures, 11 flaky |
| 6 | every row that does not reproduce is named, with a written reason |
| 7 | all 10 build scripts ship in `pipeline/` |
| 8 | none of them carries a token, a key or a build-machine path |
| 9 | the published dataset really holds 17,000 rows and 6 columns, from `datasets-server.huggingface.co` |
| 10 | every column is populated on all 17,000 rows, zero nulls and zero blanks |
| 11 | the `source` column's seven literal values and their counts, read live off the parquet index |
| 12 | filtering `source == "Dolly-15k"` on the published parquet leaves exactly 16,368 rows |
| 13 | the verification manifest downloads, 7,127,921 bytes, 13,260 entries, expected SHA-256 |
| 14 | manifest entries hash to the rows they claim: 228 sampled rows re-hashed, zero mismatches |
| 15 | the recheck evidence here re-derives exactly from the live manifest |
| 16 | the 10 scripts in `pipeline/` are byte identical to the copies published with the dataset |
| 17 | the weights download, HTTP 200, and their exact byte size |
| 18 | the base model repository resolves, HTTP 200 |
| 19 | the adapter targets `q_proj` and `v_proj` and excludes nothing |
| 20 | but `v_proj` landed on only 50 of 60 layers, re-derived from the safetensors header |
| 21 | the training run: 59 steps, peak gradient norm 677.21, 29 steps above the clip threshold |
| 22 | the Kaggle mirror answers HTTP 200 to a logged out fetch |
| 23 | both live interfaces answer HTTP 200 |

Check 20 reads the safetensors header with an HTTP range request, so it re-derives the tensor
facts in about a second instead of downloading 43 MB.

## The numbers

| | |
|---|---:|
| Rows | 17,000 |
| Columns, all populated on all rows | 6 |
| Independently verified | 13,260 (78.0%) |
| Not verified, general ballast | 3,740 (22.0%) |
| Math, checked against an independent gold answer | 9,104 |
| Code, executed against its unit tests | 4,156 |
| Verified rows that do not reproduce on a re-run | 21 (0.16%) |
| Source corpora, all public and named | 7 |
| Rows with a missing value in any column | 0 |
| Attribution-only subset, share-alike rows dropped | 16,368 |

Difficulty on the verified rows: 6,242 hard, 7,018 medium. Every math row terminates in a
`\boxed{}` answer, all 9,104 of them, which is the contract to grade against. Every code row is a
single fenced Python block that parses, all 4,156.

The dataset was decontaminated by 13-gram overlap against the genuine held-out splits: GSM8K test
(1,319), MBPP test (500), HumanEval (164). MBPP and HumanEval were used only as decontamination
targets, so no row derives from either. An earlier build pointed at the *train* splits of GSM8K
and MBPP, which means the actual benchmarks had never been checked. That was found and fixed
before publication, and the fix is in `pipeline/expand_pools.py` and `pipeline/assemble_v3.py`
where you can read it.

## Provenance, all seven sources

The `source` column carries the literal strings in the middle column. Filter on those, not on the
prose names, or your filter matches nothing.

| Upstream corpus | `source` value | Rows | Publisher's licence |
|---|---|---:|---|
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | `OpenMathReasoning` | 4,380 | CC-BY-4.0 |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | `OpenCodeInstruct` | 3,770 | CC-BY-4.0 |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | `OpenMathInstruct-2` | 3,248 | CC-BY-4.0 |
| [Tulu-3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `Tulu-3` | 3,108 | ODC-BY |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | `OpenR1-Math` | 1,476 | Apache-2.0 |
| [**Dolly-15k**](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `Dolly-15k` | **632** | **CC-BY-SA-3.0** |
| [DeepMind code_contests](https://huggingface.co/datasets/deepmind/code_contests) | `code_contests` | 386 | CC-BY-4.0 |
| **Total** | | **17,000** | released as CC-BY-SA-4.0 |

**The aggregate is CC-BY-SA-4.0 and one source is the reason.** The 632 Dolly-15k rows carry
CC-BY-SA-3.0, share-alike propagates, and nothing else here is share-alike. Those 632 rows all sit
in the unverified general slice, so dropping them costs nothing on the verified side:

```python
attribution_only = ds.filter(lambda r: r["source"] != "Dolly-15k")
print(len(attribution_only))   # 16368
```

Check 12 runs that filter against the published parquet through the Hub's own index, so the 16,368
is measured rather than subtracted on this page.

## The rows that do not reproduce

Re-running the original checker over the published rows does not return all 13,260. It returns
between 13,242 and 13,244. I would rather write down which rows fail and why than round the number
up. All 21 are flagged in `extras/verification_manifest.jsonl` under `reproduces_on_recheck`, with
a reason attached, and the derived list ships here as
[`evidence/recheck_flags.json`](evidence/recheck_flags.json).

| What fails | Count | Rows | Why |
|---|---:|---|---|
| Math, every run | 9 | 1007, 3073, 6620, 9369, 12341, 12448, 13733, 13926, 15312 | Stored gold and boxed answer are the same value in different LaTeX delimiters, `\(\infty\)` against `\infty`. `verify_core.normalize` does not strip inline-math delimiters, so it calls them different answers. That is a gap in my normaliser, not wrong data, and I left the checker as it was built rather than tuning it after the fact. |
| Code, every run | 1 | 2364 | All ten upstream asserts fail deterministically on re-execution under a current Python. |
| Code, some runs | 11 | see the evidence file | The upstream unit tests are nondeterministic. Run one three times and you get 10/10, 9/10, 10/10. Six to eight fail on any given run and the set changes, which is why the code count moves and the math count does not. |

Check 15 re-downloads the manifest, re-derives that table from it, and fails if the file shipped
here disagrees by a single row or reason.

Two more claims in the shipped artifacts that did not survive re-auditing, kept here rather than
quietly dropped. `evidence/build_manifest.json` records the output contract as "math ends with
'The final answer is $\boxed{...}$.'" and the training blueprint called that contract uniform.
It is not. **1,006 of the 9,104 math rows end with that exact sentence.** What does hold on all
9,104 is that they terminate in a `\boxed{}` answer, so grade on the last `\boxed{}` and never on
the sentence. The build manifest ships byte identical to the copy published with the dataset,
overstatement included, because a judge should be able to SHA-256 the two files and get one digest.

## Reproducing the build, in order

Five stages, all in `pipeline/`, run in this order. Full detail per stage, including inputs,
outputs and what each filter drops, is in [PIPELINE.md](PIPELINE.md).

```bash
pip install -r requirements.txt
# then pull the seven public corpora into data/ as JSONL, see PIPELINE.md

# 1. candidate pools per domain. math and code are verified at this step, not filtered
python pipeline/build_math.py
python pipeline/build_code.py
python pipeline/build_general.py

# 2. quality gate over the pools, with a drop-reason tally
python pipeline/harden.py

# 3. more verified rows on top, at the same bar, decontaminated against the real held-out splits
python pipeline/expand_pools.py --domain math    --target 6000
python pipeline/expand_pools.py --domain code    --target 1500
python pipeline/expand_pools.py --domain general --target 1200

# 4. final 17,000 rows, output contract enforced, dedup, build_manifest.json written
python pipeline/assemble_v3.py --target 17000

# 5. third-party re-check of the PUBLISHED rows, rebuilds nothing
pip install pyarrow sympy
python pipeline/verify_published.py
```

Stage 5 is the one to run if you only run one thing. It downloads the published parquet and the
published verification manifest, then re-applies the original two checks to the rows exactly as
shipped. On eight workers it takes about a minute.

`pipeline/build_code.py` and `pipeline/verify_published.py` execute untrusted third-party code
from public corpora. Each candidate runs in its own short-lived subprocess with a hard timeout,
but a subprocess is not a sandbox. Run them in a container or a throwaway VM.

`verify.py` needs none of this. It is standard library only so that checking the claims never
requires trusting the dependency tree that produced them.

## The training result, straight

I ran the full corpus through Adaption AutoScientist on `google/gemma-4-31B-it`. 17,586 rows were
ingested.

| | |
|---|---|
| Base model | `google/gemma-4-31B-it` |
| Trained model | `adaption_gemma_4_31b_it_verified_math_code_instr_3ce14a29` |
| Win rate, adapted against base | **46 against 54** |
| Optimizer steps | 59 |
| Clip threshold, `max_grad_norm` | 2 |
| Peak gradient norm | **677.21** at step 38 |
| Median gradient norm | 1.97 |
| Steps above the clip threshold | **29 of 59** |
| Steps above 1.0 | 39 of 59 |

**The adapted model scored below its base. It went backwards.** There is no way to read 46 against
54 as anything else, and the rest of this section is the mechanism from the telemetry rather than
a guess. Every gradient figure above is a value in
[`trainer_state.json`](https://huggingface.co/manifesta/adaption_verified_math_code_instruct/blob/main/trainer_state.json)
in the model repo, and check 21 re-derives all of them from that file.

**59 optimizer steps is not a fine-tune, it is a nudge.** At that step count the model barely
moved from where it started, so most of what the eval measured was still the base model.

**A median gradient norm of 1.97 against a clip threshold of 2 means half the run sat on the
threshold, and 29 of the 59 steps went over it.** Clipping keeps the direction and throws the
magnitude away, so on roughly half the steps the optimizer took a fixed-size step along whatever
direction that batch pointed in, with the relative weighting between batches gone. A peak of
677.21 is two orders of magnitude past the threshold. For comparison, a companion run on the same
platform with the same clip threshold and a different dataset peaked at 0.72 and never clipped
once. That is the signature of a learning rate or warmup mismatch for this base and batch shape.
A short run that also takes one enormous step is a short run that has been actively damaged.

**The base was already strong here.** Competition mathematics with `\boxed{}` answers and Python
with unit tests are exactly the domains a frontier instruction tune spends its budget on. There
was little headroom for a 17k-row supervised pass to claim. The useful finding across the runs
behind this entry was consistent: win rate tracked how weak the base model was in the target
domain, not how large or how clean the dataset was.

### One more thing in the adapter, and it is checkable

`adapter_config.json` asks for LoRA on `q_proj` and `v_proj`, r=8, alpha=8, with an empty
`exclude_modules` list. So the config asks for both projections on every layer. The shipped
tensors disagree.

**`q_proj` landed on all 60 layers. `v_proj` landed on 50.** The ten it missed are layers 5, 11,
17, 23, 29, 35, 41, 47, 53 and 59, which is every sixth layer starting at 5. Those are the
`full_attention` layers. This base sets `attention_k_eq_v`, so on the global-attention layers K and
V share a projection and there is no separate `v_proj` module for PEFT to wrap. Nothing declares
this anywhere in the config. It is only visible in the tensor names.

Check 20 re-derives it from the live file in about a second: it reads the first eight bytes of
`adapter_model.safetensors` with a range request to get the header length, pulls just the header
with a second range request, counts the 220 tensor keys, and asserts the exact set of missing
layers and that they are evenly spaced six apart. No 43 MB download.

That is not the reason the run went backwards, the gradient norms are. But if you adapt this base
and assume your value projections are uniform across depth, they are not.

### Adaptive Data, the data quality pass

| Metric | Before | After |
|---|---|---|
| Quality score | 8.0 | 8.2 |
| Grade | B | B |
| Percentile | 15.3 | 17.8 |

A small lift and the grade did not move. The corpus was already near the top of what that scorer
rewards before the pass ran, which is what verified data rather than filtered data should look
like going in.

Use this dataset when your base model is weak at verified mathematics and executable Python, or
when you need mechanically checkable targets for evaluation and reward modelling. Do not expect it
to beat a large, already strong instruction-tuned model on those same domains in 59 steps. The
artifact worth taking from this entry is the corpus and its audit trail, not the checkpoint.

## Limitations

- **3,740 rows are not verified.** They are anti-forgetting ballast, deliberately. Filter
  `verify_level != "-"` for the 13,260 that were checked.
- **21 of those 13,260 do not survive a re-run**, listed above with reasons.
- **Verification proves the answer, not the reasoning.** A math row is kept when its final answer
  matches gold. A solution could in principle reach a correct answer by a flawed route.
- **Code verification is unit-test verification.** Solutions pass the tests shipped with them.
  Test suites vary in thoroughness, and passing them is not proof of general correctness.
- **1,606 prompts are not in English**, 1,596 of them from Tulu-3 and 1,601 of them in the
  unverified general slice. The verified math and code slices carry 5 between them.
- **26 near-duplicate prompt pairs survived**, 49 rows, top Jaccard 0.765. The MinHashLSH index
  used during the build reports none of them because it is approximate. An exhaustive exact-Jaccard
  sweep finds them.
- **Heavily weighted toward competition mathematics and Python.** Not a general
  programming-language corpus.
- **No human review at scale.** Correctness here is mechanical. Samples were read by hand, 17,000
  rows were not.
- **The published training result is a regression.** Reported above with the diagnosis.

## What is in this repository

```
verify.py                     23 checks, standard library only, the entry point
derive_recheck_flags.py       rebuilds evidence/recheck_flags.json from the published manifest
evidence/
  build_manifest.json         the build's own output record, byte identical to the published copy
  recheck_flags.json          every verified row that does not reproduce, with the reason
pipeline/
  common.py                   JSONL IO, prompt normalisation, n-gram decontaminator
  build_math.py               1. math pool, kept only if the answer matches an independent gold
  build_code.py               1. code pool, kept only if every shipped unit test passes on execution
  build_general.py            1. the unverified general slice
  verify_core.py              math answer equivalence: string, then numeric, then symbolic
  verify_code.py              code verification by execution in a fresh subprocess
  harden.py                   2. quality gate with a drop-reason tally
  expand_pools.py             3. more verified rows, decontaminated against the real held-out splits
  assemble_v3.py              4. final 17,000 rows, contract enforced, manifest written
  verify_published.py         5. third-party re-check of the published rows
PIPELINE.md                   per-stage inputs, outputs and commands
requirements.txt              for the pipeline; verify.py needs none of it
LICENSE                       CC BY-SA 4.0, with why the aggregate is share-alike
.github/workflows/verify.yml  runs verify.py on push, on PR, and daily
```

## Links

- Dataset: <https://huggingface.co/datasets/manifesta/verified-math-code-17k>
- Weights: <https://huggingface.co/manifesta/adaption_verified_math_code_instruct>
- Kaggle mirror: <https://www.kaggle.com/datasets/aivarasnavardauskas/verified-math-code-17k>
- Live interfaces: <https://manifesta.adaptionlabs.app/> and <https://demo-theta-one-40.vercel.app>
- Verification manifest, 13,260 entries:
  [`extras/verification_manifest.jsonl`](https://huggingface.co/datasets/manifesta/verified-math-code-17k/blob/main/extras/verification_manifest.jsonl)
- Built with Adaptive Data and AutoScientist by [Adaption](https://adaptionlabs.ai)

## Citation

```bibtex
@misc{verified_math_code_17k,
  title  = {Verified Math \& Code: 17,000 execution- and gold-checked reasoning rows},
  author = {Aivaras Navardauskas},
  year   = {2026},
  url    = {https://huggingface.co/datasets/manifesta/verified-math-code-17k}
}
```

## Licence

CC BY-SA 4.0, see [LICENSE](LICENSE). The aggregate is share-alike because of the 632 Dolly-15k
rows. Rows remain subject to the terms of the upstream corpora listed in the provenance table.
Credit those publishers alongside this release.
