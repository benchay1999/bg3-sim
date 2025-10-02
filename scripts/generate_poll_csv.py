#!/usr/bin/env python3
"""
Generate a CSV for a multiple-choice poll from a JSONL inference file.

Each JSONL line is expected to include:
- "id": str
- "context": str (path to a JSON file that contains a "context" field)
- "conversation": str
- "character": str
- "ground_truth_category": str

The script resolves each line's context path, loads that JSON, extracts its
"context" string, and writes a CSV with the following columns:

- id
- context
- conversation
- character
- ground_truth_category
- question (context + three newlines + conversation)
- option_1 (Highly approve)
- option_2 (Approve)
- option_3 (Neutral)
- option_4 (Disapprove)
- option_5 (Highly disapprove)

Usage:
  python scripts/generate_poll_csv.py \
      /home/wschay/bg3-sim/test/1002_gpt-4o-mini_astarion_llm_approvals.jsonl \
      /home/wschay/bg3-sim/test/astarion_poll.csv

Optional flags:
  --base-dir BASE_DIR   Base directory for resolving relative context paths
                        (default: the repository root / current working dir).
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional


CSV_FIELDNAMES = [
    "id",
    "context",
    "conversation",
    "character",
    "ground_truth_category",
    "question",
    "option_1",
    "option_2",
    "option_3",
    "option_4",
    "option_5",
]


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def resolve_context_path(context_path: str, base_dir: str) -> Optional[str]:
    # Trim an optional leading '@' if present.
    if context_path.startswith("@"): 
        context_path = context_path[1:]

    # Absolute path as-is
    if os.path.isabs(context_path) and os.path.exists(context_path):
        return context_path

    # Try base_dir + context_path
    candidate = os.path.join(base_dir, context_path)
    if os.path.exists(candidate):
        return candidate

    # Try resolving relative to current working directory as fallback
    candidate = os.path.abspath(context_path)
    if os.path.exists(candidate):
        return candidate

    return None


def extract_context_text(context_json_path: str) -> str:
    content = read_json(context_json_path)
    if not content:
        return ""
    value = content.get("context")
    if isinstance(value, str):
        return value
    return ""


def process_jsonl(input_jsonl: str, base_dir: str, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:

        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            item_id = row.get("id", "")
            context_path = row.get("context", "")
            conversation = row.get("conversation", "")
            character = row.get("character", "")
            ground_truth_category = row.get("ground_truth_category", "")

            resolved_context_path = (
                resolve_context_path(str(context_path), base_dir) if context_path else None
            )
            context_text = extract_context_text(resolved_context_path) if resolved_context_path else ""

            question = f"{context_text}\n\n\n{conversation}"

            writer.writerow({
                "id": item_id,
                "question": question,
                "option_1": "Highly approve",
                "option_2": "Approve",
                "option_3": "Neutral",
                "option_4": "Disapprove",
                "option_5": "Highly disapprove",
                "character": character,
                "context": context_text,
                "conversation": conversation,
                "ground_truth_category": ground_truth_category,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poll CSV from JSONL with external context JSONs.")
    parser.add_argument("input_jsonl", help="Path to the JSONL inference file")
    parser.add_argument("output_csv", help="Path to write the CSV output")
    parser.add_argument(
        "--base-dir",
        default=os.getcwd(),
        help="Base directory for resolving relative context paths (default: current working dir)",
    )
    args = parser.parse_args()

    process_jsonl(args.input_jsonl, args.base_dir, args.output_csv)


if __name__ == "__main__":
    main()


