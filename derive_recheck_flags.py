#!/usr/bin/env python3
"""
derive_recheck_flags.py - rebuild evidence/recheck_flags.json from the published manifest.

evidence/recheck_flags.json is not a file I wrote by hand. It is a pure function of
extras/verification_manifest.jsonl as published with the dataset: every verified row whose
`reproduces_on_recheck` is not `true`, with the reason the manifest gives.

verify.py check 15 runs this same derivation against the live manifest and fails if the
result differs from the file committed here, so the shipped copy cannot drift away from the
published one. Run this script only to regenerate after the manifest itself changes.

    python derive_recheck_flags.py            # rewrite evidence/recheck_flags.json
    python derive_recheck_flags.py --stdout   # print it instead

Standard library only, like verify.py.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import urllib.request

MANIFEST_URL = ("https://huggingface.co/datasets/manifesta/verified-math-code-17k"
                "/resolve/main/extras/verification_manifest.jsonl")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "evidence", "recheck_flags.json")

COMMENT = [
    "Derived from extras/verification_manifest.jsonl as published with the dataset.",
    "It records every verified row that does NOT come back clean when the original",
    "checker is re-run over the published parquet. verify.py re-downloads the manifest,",
    "re-derives this file from it, and fails if the two disagree.",
    "Regenerate with: python derive_recheck_flags.py",
]


def derive(raw: bytes) -> dict:
    """Manifest bytes in, evidence dict out. Deterministic, no clock, no environment."""
    entries = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    flagged = sorted((e for e in entries if e.get("reproduces_on_recheck") is not True),
                     key=lambda e: e["row"])
    by_note: dict[str, list[int]] = {}
    for e in flagged:
        by_note.setdefault(e["recheck_note"], []).append(e["row"])
    return {
        "_comment": COMMENT,
        "manifest": {
            "path": "extras/verification_manifest.jsonl",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "entries": len(entries),
        },
        "totals": {
            "verified_rows": len(entries),
            "reproduce_clean": sum(1 for e in entries if e.get("reproduces_on_recheck") is True),
            "flagged": len(flagged),
            "deterministic_failures": sum(1 for e in flagged
                                          if e.get("reproduces_on_recheck") is False),
            "flaky": sum(1 for e in flagged if e.get("reproduces_on_recheck") == "flaky"),
        },
        "by_domain": dict(collections.Counter(e["domain"] for e in flagged)),
        "reasons": [
            {"note": note, "rows": sorted(rows), "count": len(rows)}
            for note, rows in sorted(by_note.items(), key=lambda kv: -len(kv[1]))
        ],
        "rows": [
            {"row": e["row"], "domain": e["domain"], "source": e["source"],
             "reproduces_on_recheck": e["reproduces_on_recheck"],
             "recheck_note": e["recheck_note"]}
            for e in flagged
        ],
    }


def render(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="local manifest file, else it is downloaded")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    args = ap.parse_args()

    if args.manifest:
        raw = open(args.manifest, "rb").read()
    else:
        print(f"downloading {MANIFEST_URL}", file=sys.stderr)
        req = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "derive/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()

    text = render(derive(raw))
    if args.stdout:
        sys.stdout.write(text)
    else:
        with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"wrote {OUT} ({len(text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
