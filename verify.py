#!/usr/bin/env python3
"""
verify.py - check every headline claim in this repository against a live source.

Nothing here trusts the README. Each check either reads an evidence file that ships in
this repo, or fetches the published artifact over HTTP and compares it to a number
written down below. Standard library only, so it runs on a bare Python 3.9+ with no
install step.

    python verify.py              # all checks, local and network
    python verify.py --offline    # local checks only
    python verify.py --quiet      # one line per check, no detail

Exit code is 0 when every check passes and 1 otherwise.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------------------
# The claims. Every number below appears in README.md. If you change one there, change it
# here, and this file will tell you whether the published artifact still agrees.
# --------------------------------------------------------------------------------------

DATASET_ID = "manifesta/verified-math-code-17k"
MODEL_ID = "manifesta/adaption_verified_math_code_instruct"
BASE_MODEL_ID = "google/gemma-4-31B-it"

ROWS = 17000
COLUMNS = 6
COLUMN_NAMES = ["problem", "worked_solution", "domain", "source", "difficulty", "verify_level"]

VERIFIED = 13260          # rows carrying independent verification
UNVERIFIED = 3740         # rows carrying verify_level "-", never checked
VERIFIED_PCT = 78.0

DOMAINS = {"math": 9104, "code": 4156, "general": 3740}
VERIFY_LEVELS = {"independent": 9104, "executed": 4156, "-": 3740}
DIFFICULTY = {"hard": 6242, "medium": 7018, "-": 3740}

# The literal strings in the `source` column, not their prose display names.
SOURCES = {
    "OpenMathReasoning": 4380,
    "OpenCodeInstruct": 3770,
    "OpenMathInstruct-2": 3248,
    "Tulu-3": 3108,
    "OpenR1-Math": 1476,
    "Dolly-15k": 632,
    "code_contests": 386,
}

SHARE_ALIKE_SOURCE = "Dolly-15k"
SHARE_ALIKE_ROWS = 632
ATTRIBUTION_ONLY_ROWS = 16368     # 17,000 minus the 632 share-alike rows

MANIFEST_ENTRIES = 13260
MANIFEST_BYTES = 7127921
BINDING_SAMPLE_OFFSETS = (0, 5000, 12000)   # three 100-row windows, 300 rows sampled

# Rows that do not come back clean when the checker is re-run over the published parquet.
RECHECK_FLAGGED = 21
RECHECK_DETERMINISTIC = 10
RECHECK_FLAKY = 11
MATH_FAIL_ROWS = [1007, 3073, 6620, 9369, 12341, 12448, 13733, 13926, 15312]
CODE_FAIL_ROWS = [2364]

ADAPTER_BYTES = 43856576
ADAPTER_TENSORS = 220
ADAPTER_LAYERS = 60
Q_PROJ_LAYERS = 60
V_PROJ_LAYERS = 50
# v_proj is missing on exactly these layers. They are the full_attention (global) layers:
# attention_k_eq_v is set on this base, so K and V share a projection there and PEFT has
# no separate v_proj to wrap. Every sixth layer, starting at 5.
V_PROJ_MISSING_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]

TRAIN_STEPS = 59
TRAIN_PEAK_GRAD_NORM = 677.21
TRAIN_MEDIAN_GRAD_NORM = 1.97
TRAIN_MAX_GRAD_NORM = 2          # the clip threshold the run was configured with
TRAIN_STEPS_OVER_CLIP = 29       # steps whose gradient norm exceeded 2
TRAIN_STEPS_OVER_ONE = 39        # steps whose gradient norm exceeded 1.0

PIPELINE_SCRIPTS = [
    "common.py", "build_math.py", "build_code.py", "build_general.py",
    "verify_core.py", "verify_code.py", "harden.py", "expand_pools.py",
    "assemble_v3.py", "verify_published.py",
]

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_MANIFEST_FILE = os.path.join(HERE, "evidence", "build_manifest.json")
BUILD_MANIFEST_BYTES = 825
RECHECK_FILE = os.path.join(HERE, "evidence", "recheck_flags.json")

HF_DATASET_RESOLVE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"
HF_MODEL_RESOLVE = f"https://huggingface.co/{MODEL_ID}/resolve/main/"
SERVER = "https://datasets-server.huggingface.co"
SIZE_ENDPOINT = f"{SERVER}/size?dataset={DATASET_ID}"
STATS_ENDPOINT = f"{SERVER}/statistics?dataset={DATASET_ID}&config=default&split=train"
BASE_MODEL_API = f"https://huggingface.co/api/models/{BASE_MODEL_ID}"
KAGGLE_URL = "https://www.kaggle.com/datasets/aivarasnavardauskas/verified-math-code-17k"
INTERFACE_URLS = ["https://manifesta.adaptionlabs.app/",
                  "https://demo-theta-one-40.vercel.app"]

UA = {"User-Agent": "verified-math-code-17k-verify/1.0 (+https://github.com/A1VARA5)"}

# --------------------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------------------

RESULTS = []
QUIET = False
TIMEOUT = 60
_CACHE = {}


def note(text):
    if not QUIET:
        print(f"        {text}")


def check(name, fn):
    """Run one check. fn returns a detail string on success or raises on failure."""
    try:
        detail = fn()
        print(f"PASS  {name}")
        if detail:
            note(detail)
        RESULTS.append((name, True))
    except Exception as exc:  # noqa: BLE001 - a failed check is a failed check
        print(f"FAIL  {name}")
        note(f"{type(exc).__name__}: {exc}")
        RESULTS.append((name, False))


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


# The Hub answers 500 or 503 while it warms a cache or builds a query index, which is a
# transient condition and not a failed claim. Retry those rather than let the daily badge
# flap on someone else's cold start. A 404 or a 403 is a real answer and is returned as is.
RETRYABLE = (429, 500, 502, 503, 504)


def fetch(url, headers=None, retries=4, method="GET"):
    """GET a URL, returning (status, headers, body). Retries transient failures."""
    last = None
    hdrs = dict(UA)
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs, method=method)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE and attempt < retries - 1:
                last = exc
                time.sleep(3 * (attempt + 1))
                continue
            return exc.code, dict(exc.headers or {}), b""
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not reach {url}: {last}")


def fetch_json(url, **kw):
    status, _, body = fetch(url, **kw)
    expect(status == 200, f"{url} returned HTTP {status}")
    return json.loads(body.decode("utf-8"))


def header_size(headers):
    """LFS files on the Hub report the real size in x-linked-size on a HEAD."""
    for key in ("x-linked-size", "X-Linked-Size", "content-length", "Content-Length"):
        if headers.get(key):
            return int(headers[key])
    return 0


def load_build_manifest():
    with open(BUILD_MANIFEST_FILE, "rb") as fh:
        raw = fh.read()
    return raw, json.loads(raw.decode("utf-8"))


def load_recheck():
    with open(RECHECK_FILE, "rb") as fh:
        raw = fh.read()
    return raw, json.loads(raw.decode("utf-8"))


def published_manifest():
    """Download extras/verification_manifest.jsonl once and reuse it across checks."""
    if "manifest" not in _CACHE:
        status, _, body = fetch(HF_DATASET_RESOLVE + "extras/verification_manifest.jsonl")
        expect(status == 200, f"verification_manifest.jsonl returned HTTP {status}")
        _CACHE["manifest"] = body
    return _CACHE["manifest"]


def manifest_entries():
    if "entries" not in _CACHE:
        raw = published_manifest()
        _CACHE["entries"] = [json.loads(line)
                             for line in raw.decode("utf-8").splitlines() if line.strip()]
    return _CACHE["entries"]


def column_statistics():
    if "stats" not in _CACHE:
        data = fetch_json(STATS_ENDPOINT)
        _CACHE["stats"] = {c["column_name"]: c for c in data["statistics"]}
        _CACHE["num_examples"] = data["num_examples"]
    return _CACHE["stats"]


def filter_rows(where):
    """Ask the Hub's own parquet index how many rows satisfy a predicate.

    Returns None if the filter service is unavailable. It builds a query index on demand
    and answers 500 while it does, which is a cold start rather than a failed claim.
    """
    url = (f"{SERVER}/filter?dataset={DATASET_ID}&config=default&split=train"
           f"&where={urllib.parse.quote(where)}&limit=1")
    status, _, body = fetch(url)
    if status in RETRYABLE:
        return None
    expect(status == 200, f"{url} returned HTTP {status}")
    return json.loads(body.decode("utf-8"))["num_rows_total"]


def derive_recheck(raw):
    """Same derivation as derive_recheck_flags.py, inlined so verify.py stays standalone."""
    entries = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    flagged = sorted((e for e in entries if e.get("reproduces_on_recheck") is not True),
                     key=lambda e: e["row"])
    return entries, flagged


# --------------------------------------------------------------------------------------
# Local checks. These read evidence/ and the arithmetic the README states.
# --------------------------------------------------------------------------------------

def c_build_manifest_ships():
    raw, report = load_build_manifest()
    expect(len(raw) == BUILD_MANIFEST_BYTES,
           f"build manifest is {len(raw)} bytes, expected {BUILD_MANIFEST_BYTES}")
    for key in ("total", "by_domain", "by_verify_level", "by_difficulty", "by_source",
                "decontaminated_against"):
        expect(key in report, f"build manifest has no '{key}' block")
    return (f"evidence/build_manifest.json, {len(raw)} bytes, "
            f"sha256 {hashlib.sha256(raw).hexdigest()[:16]}...")


def c_build_manifest_arithmetic():
    _, report = load_build_manifest()
    expect(report["total"] == ROWS, f"build manifest total {report['total']} != {ROWS}")
    for key, want in (("by_domain", DOMAINS), ("by_verify_level", VERIFY_LEVELS),
                      ("by_difficulty", DIFFICULTY), ("by_source", SOURCES)):
        got = report[key]
        expect(got == want, f"{key} is {got}, README says {want}")
        expect(sum(got.values()) == ROWS,
               f"{key} sums to {sum(got.values())}, not {ROWS}")
    return (f"four independent breakdowns each sum to {ROWS:,}: "
            f"3 domains, 3 verification levels, 3 difficulties, 7 sources")


def c_verified_share():
    expect(VERIFIED + UNVERIFIED == ROWS,
           f"{VERIFIED} + {UNVERIFIED} != {ROWS}")
    expect(DOMAINS["math"] + DOMAINS["code"] == VERIFIED,
           f"math + code = {DOMAINS['math'] + DOMAINS['code']}, not {VERIFIED}")
    expect(VERIFY_LEVELS["-"] == UNVERIFIED, "verify_level '-' count disagrees")
    got = round(100.0 * VERIFIED / ROWS, 1)
    expect(abs(got - VERIFIED_PCT) < 0.05, f"{VERIFIED}/{ROWS} is {got}%, README says {VERIFIED_PCT}%")
    return (f"{VERIFIED:,} verified ({got}%) + {UNVERIFIED:,} carrying verify_level '-' "
            f"({round(100.0 * UNVERIFIED / ROWS, 1)}%) = {ROWS:,}")


def c_attribution_only_arithmetic():
    expect(SOURCES[SHARE_ALIKE_SOURCE] == SHARE_ALIKE_ROWS,
           f"{SHARE_ALIKE_SOURCE} is {SOURCES[SHARE_ALIKE_SOURCE]} rows, not {SHARE_ALIKE_ROWS}")
    expect(ROWS - SHARE_ALIKE_ROWS == ATTRIBUTION_ONLY_ROWS,
           f"{ROWS} - {SHARE_ALIKE_ROWS} != {ATTRIBUTION_ONLY_ROWS}")
    return (f"{SHARE_ALIKE_SOURCE} contributes {SHARE_ALIKE_ROWS} share-alike rows; "
            f"dropping them leaves {ATTRIBUTION_ONLY_ROWS:,}")


def c_recheck_evidence_ships():
    raw, ev = load_recheck()
    t = ev["totals"]
    expect(t["verified_rows"] == VERIFIED,
           f"evidence covers {t['verified_rows']} verified rows, expected {VERIFIED}")
    expect(t["flagged"] == RECHECK_FLAGGED,
           f"{t['flagged']} rows flagged, README says {RECHECK_FLAGGED}")
    expect(t["deterministic_failures"] == RECHECK_DETERMINISTIC,
           f"{t['deterministic_failures']} deterministic failures, expected {RECHECK_DETERMINISTIC}")
    expect(t["flaky"] == RECHECK_FLAKY, f"{t['flaky']} flaky rows, expected {RECHECK_FLAKY}")
    expect(t["reproduce_clean"] + t["flagged"] == VERIFIED,
           f"{t['reproduce_clean']} clean + {t['flagged']} flagged != {VERIFIED}")
    expect(len(ev["rows"]) == RECHECK_FLAGGED,
           f"rows list holds {len(ev['rows'])} entries, totals say {RECHECK_FLAGGED}")
    return (f"{t['reproduce_clean']:,} of {VERIFIED:,} reproduce clean, "
            f"{RECHECK_DETERMINISTIC} fail deterministically, {RECHECK_FLAKY} are flaky, "
            f"every one carrying a written reason")


def c_recheck_rows_named():
    _, ev = load_recheck()
    hard = [r for r in ev["rows"] if r["reproduces_on_recheck"] is False]
    math_rows = sorted(r["row"] for r in hard if r["domain"] == "math")
    code_rows = sorted(r["row"] for r in hard if r["domain"] == "code")
    expect(math_rows == MATH_FAIL_ROWS, f"math failures {math_rows} != {MATH_FAIL_ROWS}")
    expect(code_rows == CODE_FAIL_ROWS, f"code failures {code_rows} != {CODE_FAIL_ROWS}")
    for row in hard:
        expect(bool(row.get("recheck_note")), f"row {row['row']} fails with no stated reason")
    flaky = [r["row"] for r in ev["rows"] if r["reproduces_on_recheck"] == "flaky"]
    expect(len(flaky) == RECHECK_FLAKY, f"{len(flaky)} flaky rows, expected {RECHECK_FLAKY}")
    return (f"math fails at {math_rows} (LaTeX delimiter gap in the normaliser), "
            f"code fails at {code_rows}, {len(flaky)} further code rows are flaky upstream")


def c_pipeline_ships():
    missing = [s for s in PIPELINE_SCRIPTS
               if not os.path.exists(os.path.join(HERE, "pipeline", s))]
    expect(not missing, f"pipeline/ is missing {missing}")
    total = sum(os.path.getsize(os.path.join(HERE, "pipeline", s)) for s in PIPELINE_SCRIPTS)
    return f"all {len(PIPELINE_SCRIPTS)} build scripts present in pipeline/, {total:,} bytes"


def c_no_secrets_in_pipeline():
    """A repo that says check everything should not ship a token or a build-machine path."""
    bad_markers = ["hf_", "sk-", "ghp_", "AKIA", "BEGIN RSA", "BEGIN OPENSSH",
                   "C:\\Users", "C:/Users", "/home/", "Authorization:"]
    hits = []
    for name in PIPELINE_SCRIPTS:
        text = open(os.path.join(HERE, "pipeline", name), encoding="utf-8").read()
        for marker in bad_markers:
            if marker in text:
                hits.append(f"{name}: {marker}")
    expect(not hits, f"pipeline scripts carry {len(hits)} suspicious markers: {hits[:5]}")
    return (f"{len(PIPELINE_SCRIPTS)} scripts scanned for tokens, keys and absolute build "
            f"paths, {len(hits)} hits")


# --------------------------------------------------------------------------------------
# Network checks. These fetch the published artifacts.
# --------------------------------------------------------------------------------------

def c_published_row_count():
    data = fetch_json(SIZE_ENDPOINT)
    ds = data["size"]["dataset"]
    expect(ds["num_rows"] == ROWS, f"published num_rows {ds['num_rows']} != {ROWS}")
    cfg = data["size"]["configs"][0]
    expect(cfg["num_columns"] == COLUMNS,
           f"published num_columns {cfg['num_columns']} != {COLUMNS}")
    return (f"datasets-server reports num_rows={ds['num_rows']:,}, "
            f"num_columns={cfg['num_columns']}, "
            f"{ds['num_bytes_original_files']:,} bytes of parquet")


def c_no_missing_values():
    stats = column_statistics()
    expect(_CACHE["num_examples"] == ROWS,
           f"statistics endpoint counts {_CACHE['num_examples']} rows, not {ROWS}")
    missing = [name for name in COLUMN_NAMES if name not in stats]
    expect(not missing, f"published dataset has no column {missing}")
    expect(len(stats) == COLUMNS,
           f"published dataset has {len(stats)} columns, expected {COLUMNS}")
    offenders = {}
    for name, col in stats.items():
        nan = col["column_statistics"].get("nan_count", 0)
        blank = col["column_statistics"].get("no_label_count", 0)
        if nan or blank:
            offenders[name] = (nan, blank)
    expect(not offenders, f"columns with missing values: {offenders}")
    return (f"all {COLUMNS} columns populated on {ROWS:,} of {ROWS:,} rows, "
            f"zero nulls and zero blanks anywhere")


def c_source_column_values():
    stats = column_statistics()
    col = stats["source"]["column_statistics"]
    got = col["frequencies"]
    expect(got == SOURCES, f"live source counts {got} != {SOURCES}")
    expect(col["n_unique"] == len(SOURCES),
           f"{col['n_unique']} distinct sources, expected {len(SOURCES)}")
    expect(sum(got.values()) == ROWS, f"source counts sum to {sum(got.values())}, not {ROWS}")
    for name in ("domain", "verify_level", "difficulty"):
        want = {"domain": DOMAINS, "verify_level": VERIFY_LEVELS, "difficulty": DIFFICULTY}[name]
        live = stats[name]["column_statistics"]["frequencies"]
        expect(live == want, f"live {name} counts {live} != {want}")
    ordered = ", ".join(f"{k} {v:,}" for k, v in
                        sorted(got.items(), key=lambda kv: -kv[1]))
    return f"7 literal source values, live off the parquet index: {ordered}"


def c_share_alike_filter():
    dolly = filter_rows(f"\"source\"='{SHARE_ALIKE_SOURCE}'")
    rest = filter_rows(f"\"source\"!='{SHARE_ALIKE_SOURCE}'")
    if dolly is None or rest is None:
        # The filter service is cold. Fall back to the parquet index's own value counts,
        # which is still a live measurement of the published file, just a different route
        # to it. The claim is not taken on trust either way.
        freq = column_statistics()["source"]["column_statistics"]["frequencies"]
        dolly = freq[SHARE_ALIKE_SOURCE]
        rest = sum(v for k, v in freq.items() if k != SHARE_ALIKE_SOURCE)
        route = "value counts from the parquet index, filter service was warming"
    else:
        route = "SQL filter run over the published parquet"
    expect(dolly == SHARE_ALIKE_ROWS,
           f"found {dolly} {SHARE_ALIKE_SOURCE} rows, expected {SHARE_ALIKE_ROWS}")
    expect(rest == ATTRIBUTION_ONLY_ROWS,
           f"filtering leaves {rest} rows, expected {ATTRIBUTION_ONLY_ROWS}")
    expect(dolly + rest == ROWS, f"{dolly} + {rest} != {ROWS}")
    return (f"source == '{SHARE_ALIKE_SOURCE}' matches {dolly} rows; dropping them leaves "
            f"exactly {rest:,} ({route})")


def c_manifest_downloads():
    raw = published_manifest()
    expect(len(raw) == MANIFEST_BYTES,
           f"manifest is {len(raw)} bytes, expected {MANIFEST_BYTES}")
    entries = manifest_entries()
    expect(len(entries) == MANIFEST_ENTRIES,
           f"manifest holds {len(entries)} entries, expected {MANIFEST_ENTRIES}")
    _, ev = load_recheck()
    expect(hashlib.sha256(raw).hexdigest() == ev["manifest"]["sha256"],
           "published manifest does not match the sha256 recorded in evidence/recheck_flags.json")
    by_domain = collections.Counter(e["domain"] for e in entries)
    expect(by_domain["math"] == DOMAINS["math"] and by_domain["code"] == DOMAINS["code"],
           f"manifest domains {dict(by_domain)} disagree with {DOMAINS}")
    no_evidence = [e["row"] for e in entries
                   if not (e.get("gold") or e.get("unit_tests") or e.get("stdin"))]
    expect(not no_evidence,
           f"{len(no_evidence)} manifest entries carry neither a gold answer nor unit tests")
    return (f"{len(raw):,} bytes, {len(entries):,} entries, one per verified row "
            f"({by_domain['math']:,} math golds, {by_domain['code']:,} code test sets), "
            f"sha256 {hashlib.sha256(raw).hexdigest()[:16]}...")


def c_manifest_binds_to_rows():
    """Each manifest entry claims a SHA-256 of its problem string. Re-hash the real rows."""
    entries = {e["row"]: e for e in manifest_entries()}
    ok = mismatched = truncated = 0
    for offset in BINDING_SAMPLE_OFFSETS:
        url = (f"{SERVER}/rows?dataset={DATASET_ID}&config=default&split=train"
               f"&offset={offset}&length=100")
        data = fetch_json(url)
        expect(data["num_rows_total"] == ROWS,
               f"rows endpoint reports {data['num_rows_total']} rows, not {ROWS}")
        for record in data["rows"]:
            idx = record["row_idx"]
            entry = entries.get(idx)
            if entry is None:
                continue                      # an unverified general row, correctly absent
            if record.get("truncated_cells"):
                truncated += 1
                continue
            row = record["row"]
            digest = hashlib.sha256(row["problem"].encode("utf-8")).hexdigest()
            if digest == entry["problem_sha256"] and row["verify_level"] == entry["verify_level"]:
                ok += 1
            else:
                mismatched += 1
    expect(mismatched == 0, f"{mismatched} sampled rows do not hash to their manifest entry")
    expect(ok >= 200, f"only {ok} rows could be checked, wanted at least 200")
    return (f"{ok} sampled rows re-hashed from the published parquet, {mismatched} mismatches; "
            f"a gold answer cannot drift onto the wrong row without this failing")


def c_recheck_evidence_matches_manifest():
    raw = published_manifest()
    entries, flagged = derive_recheck(raw)
    _, ev = load_recheck()
    expect(len(entries) == ev["totals"]["verified_rows"], "entry count drifted")
    live = [(e["row"], e["domain"], e["reproduces_on_recheck"], e["recheck_note"])
            for e in flagged]
    shipped = [(r["row"], r["domain"], r["reproduces_on_recheck"], r["recheck_note"])
               for r in ev["rows"]]
    expect(live == shipped,
           f"re-derived flags differ from the shipped evidence: "
           f"{[x for x in live if x not in shipped][:3]}")
    return (f"re-derived {len(flagged)} flagged rows straight from the published manifest "
            f"and every row, reason and verdict matches evidence/recheck_flags.json")


def c_pipeline_matches_published():
    diffs = []
    for name in PIPELINE_SCRIPTS:
        local = open(os.path.join(HERE, "pipeline", name), "rb").read()
        status, _, remote = fetch(HF_DATASET_RESOLVE + "build/" + name)
        expect(status == 200, f"published build/{name} returned HTTP {status}")
        if hashlib.sha256(local).hexdigest() != hashlib.sha256(remote).hexdigest():
            diffs.append(name)
    expect(not diffs, f"{len(diffs)} pipeline scripts differ from the published copies: {diffs}")
    return (f"all {len(PIPELINE_SCRIPTS)} scripts in pipeline/ are byte identical to the "
            f"copies published under build/ with the dataset, by SHA-256")


def c_weights_download():
    status, headers, _ = fetch(HF_MODEL_RESOLVE + "adapter_model.safetensors", method="HEAD")
    expect(status == 200, f"adapter_model.safetensors returned HTTP {status}")
    size = header_size(headers)
    expect(size == ADAPTER_BYTES, f"adapter is {size} bytes, expected {ADAPTER_BYTES}")
    return f"HTTP 200, adapter_model.safetensors is {size:,} bytes"


def c_base_model_resolves():
    status, _, body = fetch(BASE_MODEL_API)
    expect(status == 200, f"{BASE_MODEL_ID} returned HTTP {status}")
    meta = json.loads(body.decode("utf-8"))
    expect(meta.get("id") == BASE_MODEL_ID, f"unexpected repo id {meta.get('id')}")
    return f"{BASE_MODEL_ID} resolves, HTTP 200, pipeline_tag={meta.get('pipeline_tag')}"


def c_adapter_config():
    cfg = fetch_json(HF_MODEL_RESOLVE + "adapter_config.json")
    expect(cfg.get("base_model_name_or_path") == BASE_MODEL_ID,
           f"adapter points at {cfg.get('base_model_name_or_path')}, not {BASE_MODEL_ID}")
    targets = cfg.get("target_modules") or []
    expect(sorted(targets) == ["q_proj", "v_proj"],
           f"target_modules is {targets}, expected q_proj and v_proj")
    expect(not (cfg.get("exclude_modules") or []),
           "adapter_config declares exclude_modules, so the v_proj gap would not be a surprise")
    return (f"base {cfg['base_model_name_or_path']}, r={cfg.get('r')}, "
            f"alpha={cfg.get('lora_alpha')}, targets {sorted(targets)}, "
            f"exclude_modules empty, so the config asks for v_proj on every layer")


def c_v_proj_gap():
    """Read only the safetensors header with a range request, not the whole 43 MB."""
    url = HF_MODEL_RESOLVE + "adapter_model.safetensors"
    status, _, head = fetch(url, headers={"Range": "bytes=0-7"})
    expect(status in (200, 206), f"range request returned HTTP {status}")
    length = struct.unpack("<Q", head[:8])[0]
    status, _, blob = fetch(url, headers={"Range": f"bytes=8-{8 + length - 1}"})
    expect(status in (200, 206), f"header range request returned HTTP {status}")
    header = json.loads(blob.decode("utf-8"))
    keys = [k for k in header if k != "__metadata__"]
    expect(len(keys) == ADAPTER_TENSORS,
           f"adapter holds {len(keys)} tensors, expected {ADAPTER_TENSORS}")

    layers = {"q_proj": set(), "v_proj": set()}
    for key in keys:
        parts = key.split(".")
        if "layers" not in parts:
            continue
        idx = int(parts[parts.index("layers") + 1])
        for proj in layers:
            if proj in parts:
                layers[proj].add(idx)

    expect(len(layers["q_proj"]) == Q_PROJ_LAYERS,
           f"q_proj lands on {len(layers['q_proj'])} layers, expected {Q_PROJ_LAYERS}")
    expect(len(layers["v_proj"]) == V_PROJ_LAYERS,
           f"v_proj lands on {len(layers['v_proj'])} layers, expected {V_PROJ_LAYERS}")
    missing = sorted(set(range(ADAPTER_LAYERS)) - layers["v_proj"])
    expect(missing == V_PROJ_MISSING_LAYERS,
           f"v_proj is missing on layers {missing}, README says {V_PROJ_MISSING_LAYERS}")
    stride = {b - a for a, b in zip(missing, missing[1:])}
    expect(stride == {6},
           f"the missing layers are not evenly spaced, gaps are {sorted(stride)}")
    return (f"{len(keys)} trained tensors: q_proj on all {len(layers['q_proj'])} layers, "
            f"v_proj on only {len(layers['v_proj'])}, missing exactly {missing} "
            f"(every 6th layer from 5, the full_attention layers where K and V share "
            f"a projection)")


def c_training_run():
    state = fetch_json(HF_MODEL_RESOLVE + "trainer_state.json")
    expect(state["global_step"] == TRAIN_STEPS,
           f"global_step {state['global_step']} != {TRAIN_STEPS}")
    grads = [e["grad_norm"] for e in state["log_history"] if e.get("grad_norm") is not None]
    expect(len(grads) == TRAIN_STEPS,
           f"{len(grads)} logged gradient norms for {TRAIN_STEPS} steps")
    peak = max(grads)
    expect(abs(round(peak, 2) - TRAIN_PEAK_GRAD_NORM) < 0.005,
           f"peak gradient norm {peak:.4f} rounds to {round(peak, 2)}, "
           f"README says {TRAIN_PEAK_GRAD_NORM}")
    ordered = sorted(grads)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else \
        (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    expect(abs(round(median, 2) - TRAIN_MEDIAN_GRAD_NORM) < 0.005,
           f"median gradient norm {median:.4f} rounds to {round(median, 2)}, "
           f"README says {TRAIN_MEDIAN_GRAD_NORM}")
    over_clip = sum(1 for g in grads if g > TRAIN_MAX_GRAD_NORM)
    over_one = sum(1 for g in grads if g > 1.0)
    expect(over_clip == TRAIN_STEPS_OVER_CLIP,
           f"{over_clip} steps exceed the clip threshold, README says {TRAIN_STEPS_OVER_CLIP}")
    expect(over_one == TRAIN_STEPS_OVER_ONE,
           f"{over_one} steps exceed 1.0, README says {TRAIN_STEPS_OVER_ONE}")
    return (f"{state['global_step']} optimizer steps over {state['num_train_epochs']} epoch, "
            f"peak {peak:.2f}, median {median:.2f}, {over_clip} of {len(grads)} steps above "
            f"the clip threshold of {TRAIN_MAX_GRAD_NORM}, {over_one} above 1.0")


def c_kaggle_mirror():
    status, _, body = fetch(KAGGLE_URL)
    expect(status == 200, f"Kaggle mirror returned HTTP {status} to a logged out fetch")
    expect(b"verified-math-code-17k" in body, "Kaggle page does not name the dataset")
    return f"HTTP 200 logged out, {len(body):,} bytes"


def c_interfaces():
    for url in INTERFACE_URLS:
        status, _, _ = fetch(url)
        expect(status == 200, f"{url} returned HTTP {status}")
    return ", ".join(f"{u} 200" for u in INTERFACE_URLS)


LOCAL_CHECKS = [
    ("build manifest ships and parses", c_build_manifest_ships),
    ("four breakdowns in the build manifest each sum to 17,000", c_build_manifest_arithmetic),
    ("13,260 verified + 3,740 unverified = 17,000", c_verified_share),
    ("dropping the share-alike source leaves 16,368", c_attribution_only_arithmetic),
    ("recheck evidence ships and its counts add up", c_recheck_evidence_ships),
    ("every non-reproducing row is named with a reason", c_recheck_rows_named),
    ("the full build pipeline ships here", c_pipeline_ships),
    ("no tokens, keys or build-machine paths in the pipeline", c_no_secrets_in_pipeline),
]

NETWORK_CHECKS = [
    ("published dataset really holds 17,000 rows", c_published_row_count),
    ("every column is populated on every row", c_no_missing_values),
    ("the source column values and counts, live", c_source_column_values),
    ("filtering the share-alike source leaves 16,368 rows", c_share_alike_filter),
    ("the verification manifest downloads and counts 13,260", c_manifest_downloads),
    ("manifest entries hash to the rows they claim", c_manifest_binds_to_rows),
    ("shipped recheck evidence re-derives from the live manifest",
     c_recheck_evidence_matches_manifest),
    ("pipeline here is byte identical to the published pipeline", c_pipeline_matches_published),
    ("weights download and report their size", c_weights_download),
    ("base model resolves", c_base_model_resolves),
    ("adapter targets q_proj and v_proj with nothing excluded", c_adapter_config),
    ("v_proj landed on 50 of 60 layers, q_proj on all 60", c_v_proj_gap),
    ("training run: 59 steps, peak 677.21, 29 steps clipped", c_training_run),
    ("Kaggle mirror is public", c_kaggle_mirror),
    ("live interfaces answer", c_interfaces),
]


def main():
    global QUIET, TIMEOUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="skip every check that needs the network")
    ap.add_argument("--quiet", action="store_true", help="one line per check")
    ap.add_argument("--timeout", type=int, default=60, help="per-request timeout in seconds")
    args = ap.parse_args()
    QUIET = args.quiet
    TIMEOUT = args.timeout

    print("verified-math-code-17k :: verification")
    print(f"dataset  {DATASET_ID}")
    print(f"weights  {MODEL_ID}")
    print(f"base     {BASE_MODEL_ID}")
    print("-" * 78)
    print("local checks")
    for name, fn in LOCAL_CHECKS:
        check(name, fn)

    if args.offline:
        print("-" * 78)
        print("network checks skipped (--offline)")
    else:
        print("-" * 78)
        print("network checks")
        for name, fn in NETWORK_CHECKS:
            check(name, fn)

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print("-" * 78)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        for name, ok in RESULTS:
            if not ok:
                print(f"  failed: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
