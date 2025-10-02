#!/usr/bin/env python3
import json
import os
import random
import argparse
from typing import Dict, List
import hashlib

DEFAULT_INPUT = "/nfs_edlab/wschay/bg3-sim/approval-dataset/approval_dataset_with_id.jsonl"
DEFAULT_OUTPUT_DIR = "result-dataset"

# Context roots used in dataset `context` field
QA_CONTEXTS_DIR_PRIMARY = "qa-contexts-rag"
QA_CONTEXTS_DIR_ALT = "qa-context-rag"


# Category mapping
# 1) Highly Disapprove: score <= -5
# 2) Disapprove: (-5, -1]
# 3) Approve: [1, 5)
# 4) Highly Approve: score >= 5
CATEGORY_KEYS = [
    "highly_disapprove",
    "disapprove",
    "approve",
    "highly_approve",
]
CATEGORY_HUMAN = {
    "highly_disapprove": "Highly Disapprove",
    "disapprove": "Disapprove",
    "approve": "Approve",
    "highly_approve": "Highly Approve",
}


def to_number(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def categorize(score) -> str:
    s = to_number(score)
    if s != s:  # NaN check
        return ""
    if s <= -5:
        return "highly_disapprove"
    if -5 < s <= -1:
        return "disapprove"
    if 1 <= s < 5:
        return "approve"
    if s >= 5:
        return "highly_approve"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create balanced per-character subsets from the approval JSONL dataset using range-based categories."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT,
        help="Path to input combined JSONL dataset",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write subset JSON files",
    )
    parser.add_argument(
        "-c",
        "--characters",
        nargs="+",
        default=["Astarion", "Shadowheart", "Gale", "Wyll"],
        help="Character names to include (space-separated)",
    )
    parser.add_argument(
        "-p",
        "--per-bucket",
        type=int,
        default=200,
        help="Max samples per category per character",
    )
    parser.add_argument(
        "--session-max",
        type=int,
        default=20,
        help="Global maximum samples allowed from a single session (context JSON) across all categories for a character",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    return parser.parse_args()


def build_buckets(input_path: str, characters: List[str]) -> Dict[str, Dict[str, List[Dict]]]:
    by_char_category: Dict[str, Dict[str, List[Dict]]] = {
        char: {key: [] for key in CATEGORY_KEYS} for char in characters
    }
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Ensure a stable id exists (short 64-bit hex)
            if isinstance(obj, dict) and "conversation" in obj and "id" not in obj:
                try:
                    obj["id"] = hashlib.sha256(obj["conversation"].encode("utf-8")).digest()[:8].hex()
                except Exception:
                    pass
            label = obj.get("label", {})
            if not isinstance(label, dict):
                continue
            for char in characters:
                if char not in label:
                    continue
                cat = categorize(label.get(char))
                if not cat:
                    continue
                by_char_category[char][cat].append(obj)
    return by_char_category


def _session_id_from_context(context_path: str) -> str:
    if context_path.startswith(f"{QA_CONTEXTS_DIR_PRIMARY}/"):
        return context_path[len(QA_CONTEXTS_DIR_PRIMARY) + 1 :]
    if context_path.startswith(f"{QA_CONTEXTS_DIR_ALT}/"):
        return context_path[len(QA_CONTEXTS_DIR_ALT) + 1 :]
    return context_path


def sample_character(by_category: Dict[str, List[Dict]], per_bucket: int, session_max: int, used_ids: set) -> List[Dict]:
    sampled: List[Dict] = []
    session_counts: Dict[str, int] = {}
    for key in CATEGORY_KEYS:
        items = list(by_category[key])
        random.shuffle(items)
        taken_in_cat = 0
        for obj in items:
            if taken_in_cat >= per_bucket:
                break
            context_path = obj.get("context", "")
            session_id = _session_id_from_context(context_path)
            if session_counts.get(session_id, 0) >= session_max:
                continue
            # Use existing ID if present; only generate when missing.
            if "id" in obj and isinstance(obj["id"], str) and obj["id"]:
                # Preserve existing ID from input
                candidate_id = obj["id"]
                # Track it so generated IDs don't collide with it
                used_ids.add(candidate_id)
            else:
                # Generate 16-char ID and resolve collisions only for newly generated IDs
                conv = obj.get("conversation", "")
                candidate_id = hashlib.sha256(conv.encode("utf-8")).digest()[:8].hex()
                counter = 1
                while candidate_id in used_ids:
                    candidate_id = hashlib.sha256((conv + f"#{counter}").encode("utf-8")).digest()[:8].hex()
                    counter += 1
                obj["id"] = candidate_id
                used_ids.add(candidate_id)

            sampled.append(obj)
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
            taken_in_cat += 1
        print(
            f"  {CATEGORY_HUMAN[key]}: have {len(items)}, taking {taken_in_cat} (session-capped)"
        )
    return sampled


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Input dataset not found: {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print(
        f"Building subsets for {', '.join(args.characters)} from {args.input} (per-bucket={args.per_bucket}, seed={args.seed})"
    )

    by_char_category = build_buckets(args.input, args.characters)

    used_ids = set()
    for char in args.characters:
        print(f"Character: {char}")
        sampled = sample_character(by_char_category[char], args.per_bucket, args.session_max, used_ids)
        output_path = os.path.join(
            args.output_dir, f"{char.lower()}_approval_dataset_subset.json"
        )
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(sampled, out, ensure_ascii=False, indent=2)
        print(f"  Wrote {len(sampled)} samples to {output_path}")


if __name__ == "__main__":
    main()


