#!/usr/bin/env python3
import argparse
import json
import os
import re
import random
from typing import Dict, List, Tuple, Optional, Any
import hashlib


# Defaults mirror the builder script for paths
APPROVAL_DIR_DEFAULT = "/nfs_edlab/wschay/bg3-sim/approval-paths"
QA_CONTEXTS_DIR_PRIMARY = "qa-contexts-rag"
QA_CONTEXTS_DIR_ALT = "qa-context-rag"
WORKSPACE_ROOT = "."
OUTPUT_DEFAULT = "/nfs_edlab/wschay/bg3-sim/approval-dataset/approval_dataset_subset.jsonl"


APPROVAL_RE = re.compile(r"\[approval\]\s*(.+)")
CONTEXT_HEADER_RE = re.compile(r"^Context:\s*$", re.IGNORECASE)
SYNOPSIS_RE = re.compile(r"^Synopsis:\s*(.*)$", re.IGNORECASE)

# Canonicalize origin/companion names (same as builder)
ORIGIN_CANONICAL = {
    "astarion": "Astarion",
    "gale": "Gale",
    "karlach": "Karlach",
    "lae'zel": "Lae'zel",
    "laezel": "Lae'zel",
    "shadowheart": "Shadowheart",
    "wyll": "Wyll",
}


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def extract_context_from_lines(lines: List[str]) -> str:
    in_context = False
    context_lines: List[str] = []
    for line in lines:
        if not in_context:
            if CONTEXT_HEADER_RE.match(line):
                in_context = True
            continue
        if line.strip().startswith("Path "):
            break
        context_lines.append(line)
    context = "\n".join([l for l in context_lines if l.strip() != ""]).strip()
    if context:
        return context
    # Fallback to Synopsis
    for line in lines:
        m = SYNOPSIS_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def to_context_json_path(approval_file_path: str) -> Optional[str]:
    try:
        rel = os.path.relpath(approval_file_path, args.approval_dir)  # type: ignore[name-defined]
    except Exception:
        return None
    base = os.path.splitext(rel)[0]
    if base.endswith("_approval_paths"):
        base = base[: -len("_approval_paths")]
    candidate_primary = os.path.join(QA_CONTEXTS_DIR_PRIMARY, base + ".json")
    if os.path.isfile(candidate_primary):
        return candidate_primary
    candidate_alt = os.path.join(QA_CONTEXTS_DIR_ALT, base + ".json")
    if os.path.isfile(candidate_alt):
        return candidate_alt
    return candidate_primary


def compute_context_relpath(approval_file_path: str) -> str:
    ctx_abs = to_context_json_path(approval_file_path)
    if not ctx_abs:
        return ""
    try:
        return os.path.relpath(ctx_abs, WORKSPACE_ROOT)
    except Exception:
        return ctx_abs


def extract_paths(lines: List[str]) -> List[List[str]]:
    paths: List[List[str]] = []
    current: List[str] = []
    in_paths = False
    for line in lines:
        if line.strip().startswith("Path ") and line.strip().endswith(":"):
            in_paths = True
            if current:
                paths.append(current)
                current = []
            continue
        if in_paths:
            current.append(line)
    if current:
        paths.append(current)
    return paths


def parse_approval_payload(payload: str) -> Dict[str, int]:
    label: Dict[str, int] = {}
    parts = [p.strip() for p in payload.split(",") if p.strip()]
    for part in parts:
        m = re.match(r"(.+?)\s+(-?\d+)$", part)
        if not m:
            continue
        raw_name = m.group(1).strip()
        try:
            value = int(m.group(2))
        except ValueError:
            continue
        key = raw_name.lower()
        if key in ORIGIN_CANONICAL:
            name = ORIGIN_CANONICAL[key]
            label[name] = value
    return label


def normalize_convo_text(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def build_conversation_and_label(path_lines: List[str]) -> Optional[Tuple[str, Dict[str, int]]]:
    approval_positions: List[Tuple[int, Dict[str, int]]] = []
    processed_lines: List[str] = []
    for raw in path_lines:
        line = raw
        if re.match(r"^\s*::\s*(True|False)\s*$", line):
            continue
        text_part = line
        labels: Optional[Dict[str, int]] = None
        if "||" in line:
            left, right = line.split("||", 1)
            text_part = left.rstrip()
            appr_match = APPROVAL_RE.search(right)
            if appr_match:
                labels = parse_approval_payload(appr_match.group(1))
        else:
            appr_match_inline = APPROVAL_RE.search(line)
            if appr_match_inline:
                text_part = line[: appr_match_inline.start()].rstrip()
                labels = parse_approval_payload(appr_match_inline.group(1))
        processed_lines.append(text_part)
        if labels:
            approval_positions.append((len(processed_lines) - 1, labels))
    if not approval_positions:
        return None
    last_index, last_labels = approval_positions[-1]
    convo_lines = processed_lines[: last_index + 1]
    convo = "\n".join([ln for ln in convo_lines if ln is not None])
    convo = normalize_convo_text(convo)
    return convo, last_labels


def group_files_by_act(approval_dir: str, acts_filter: Optional[List[str]]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for root, _dirs, files in os.walk(approval_dir):
        rel = os.path.relpath(root, approval_dir)
        if rel == ".":
            act = "ROOT"
        else:
            act = rel.split(os.sep, 1)[0]
        if acts_filter and act not in acts_filter:
            continue
        for fname in files:
            if not fname.endswith("_approval_paths.txt"):
                continue
            fpath = os.path.join(root, fname)
            groups.setdefault(act, []).append(fpath)
    return groups


def round_robin_files(groups: Dict[str, List[str]], chunk_size: int, max_files: int) -> List[Tuple[str, str]]:
    order = list(groups.keys())
    random.shuffle(order)
    # Shuffle within each act
    for act in order:
        random.shuffle(groups[act])
    indices = {act: 0 for act in order}
    out: List[Tuple[str, str]] = []
    remaining = sum(len(groups[a]) for a in order)
    while remaining > 0 and (max_files <= 0 or len(out) < max_files):
        progressed = False
        for act in order:
            start = indices[act]
            end = min(start + chunk_size, len(groups[act]))
            if start >= end:
                continue
            for i in range(start, end):
                if max_files > 0 and len(out) >= max_files:
                    break
                out.append((act, groups[act][i]))
            indices[act] = end
            remaining = sum(len(groups[a]) - indices[a] for a in order)
            progressed = True
            if max_files > 0 and len(out) >= max_files:
                break
        if not progressed:
            break
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a round-robin per-act subset of approval paths and build a JSONL dataset."
    )
    parser.add_argument("--approval-dir", default=APPROVAL_DIR_DEFAULT, help="Root directory containing *_approval_paths.txt files.")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output JSONL path for the sampled subset.")
    parser.add_argument("--chunk-size", type=int, default=10, help="Number of files to take consecutively from one act before moving on.")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files to process in total (0 = no limit).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling acts and files.")
    parser.add_argument("--acts", nargs="*", default=None, help="Optional list of act names to include (e.g., Act1 Act1b Act2 Act3 Camp Companions).")
    parser.add_argument("--per-category", type=int, default=200, help="Number of samples to write per act category.")
    parser.add_argument("--session-max", type=int, default=20, help="Global maximum samples allowed from a single session (context JSON) across all categories.")
    return parser.parse_args()


def main() -> None:
    global args  # used by to_context_json_path for relpath against approval-dir
    args = parse_args()
    random.seed(args.seed)

    groups = group_files_by_act(args.approval_dir, args.acts)
    if not groups:
        print("No *_approval_paths.txt files found for the given filters.")
        return

    schedule = round_robin_files(groups, args.chunk_size, args.max_files)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    seen_keys = set()
    per_act_counts = {act: 0 for act in groups.keys()}  # files processed per act
    category_counts = {act: 0 for act in groups.keys()}  # samples written per act category
    session_counts: Dict[str, int] = {}
    written = 0
    used_ids = set()

    with open(args.output, "w", encoding="utf-8") as out:
        for act, fpath in schedule:
            per_act_counts[act] += 1
            # If this category already met quota, skip this file quickly
            if category_counts.get(act, 0) >= args.per_category:
                # Check if all targets are met to allow early exit
                if all(category_counts.get(a, 0) >= args.per_category for a in category_counts.keys()):
                    break
                continue
            try:
                lines = read_text(fpath)
            except Exception:
                continue
            context_rel = compute_context_relpath(fpath)
            path_blocks = extract_paths(lines)
            for path_lines in path_blocks:
                result = build_conversation_and_label(path_lines)
                if not result:
                    continue
                conversation, label = result
                if not label:
                    continue
                key = (context_rel, conversation)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Determine session id from context path (strip qa contexts prefix)
                if context_rel.startswith(f"{QA_CONTEXTS_DIR_PRIMARY}/"):
                    session_id = context_rel[len(QA_CONTEXTS_DIR_PRIMARY) + 1 :]
                elif context_rel.startswith(f"{QA_CONTEXTS_DIR_ALT}/"):
                    session_id = context_rel[len(QA_CONTEXTS_DIR_ALT) + 1 :]
                else:
                    session_id = context_rel

                # Enforce global per-session cap
                if session_counts.get(session_id, 0) >= args.session_max:
                    continue

                # Enforce per-category quota
                if category_counts.get(act, 0) >= args.per_category:
                    # Category full; skip remaining from this act
                    continue
                # Generate collision-checked short id (16 hex chars from 64-bit truncated SHA-256)
                base_id = hashlib.sha256(conversation.encode("utf-8")).digest()[:8].hex()
                candidate_id = base_id
                counter = 1
                while candidate_id in used_ids:
                    candidate_id = hashlib.sha256((conversation + f"#{counter}").encode("utf-8")).digest()[:8].hex()
                    counter += 1
                used_ids.add(candidate_id)

                obj: Dict[str, Any] = {
                    "id": candidate_id,
                    "context": context_rel,
                    "conversation": conversation,
                    "label": label,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                category_counts[act] = category_counts.get(act, 0) + 1
                session_counts[session_id] = session_counts.get(session_id, 0) + 1

                # If all categories have reached targets, stop early
                if all(category_counts.get(a, 0) >= args.per_category for a in category_counts.keys()):
                    break
            # After finishing this file, if all targets met, exit outer loop
            if all(category_counts.get(a, 0) >= args.per_category for a in category_counts.keys()):
                break

    # Summary
    acts_str = ", ".join(sorted(groups.keys()))
    print(f"Processed files per act: {per_act_counts}")
    print(f"Samples written per act: {category_counts}")
    print(f"Acts included: {acts_str}")
    print(f"Per-category target: {args.per_category}, per-session global cap: {args.session_max}")
    print(f"Unique sessions used: {len(session_counts)} (capped at {args.session_max} each)")
    print(f"Wrote {written} samples to {args.output}")


if __name__ == "__main__":
    main()


