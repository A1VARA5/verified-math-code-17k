"""Shared IO, prompt-normalization, dedup and decontamination helpers for the build."""
from __future__ import annotations
import json, re, os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(_ROOT, "data")

# POOLS / OUT are overridable via env so we can build from hardened pools without
# touching the originals (POOLS_DIR / OUT_DIR). Relative overrides resolve against
# the model root, then are made absolute so downstream path-joins stay correct.
def _resolve(env, default):
    p = os.environ.get(env) or default
    if not os.path.isabs(p):
        p = os.path.join(_ROOT, p)
    return os.path.abspath(p)

POOLS = _resolve("POOLS_DIR", "pools")
OUT = _resolve("OUT_DIR", "dataset")
os.makedirs(POOLS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def read_jsonl(name: str, limit: int | None = None):
    path = name if os.path.isabs(name) else os.path.join(DATA, name)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_think(text: str) -> str:
    """Keep the reasoning content but drop the literal R1 <think> tag tokens."""
    return text.replace("<think>", "").replace("</think>", "").strip()


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]+")

def normalize_prompt(s: str) -> str:
    s = (s or "").lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def dedup_key(prompt: str) -> str:
    return normalize_prompt(prompt)


def ngrams(text: str, n: int = 13):
    toks = normalize_prompt(text).split()
    return {" ".join(toks[i : i + n]) for i in range(0, max(0, len(toks) - n + 1))}


class Decontaminator:
    """Flags a training prompt as contaminated if it shares any n-gram with a
    held-out benchmark prompt (GSM8K / MBPP / HumanEval). 13-gram by default."""

    def __init__(self, n: int = 13):
        self.n = n
        self.bench: set[str] = set()

    def add_benchmark_text(self, text: str):
        self.bench |= ngrams(text, self.n)

    def is_contaminated(self, text: str) -> bool:
        g = ngrams(text, self.n)
        if not g:
            return False
        return not g.isdisjoint(self.bench)
