#!/usr/bin/env python3
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import random
import signal
import hashlib

APPROVAL_DIR = "/nfs_edlab/wschay/bg3-sim/approval-paths"
QA_CONTEXTS_DIR_PRIMARY = "qa-contexts-rag"
QA_CONTEXTS_DIR_ALT = "qa-context-rag"  # fallback if singular dir exists
WORKSPACE_ROOT = "/home/wschay/bg3-sim"
OUTPUT_PATH = "/nfs_edlab/wschay/bg3-sim/approval-dataset/approval_dataset_251014.jsonl"

MAX_OUTPUT_SIZE_BYTES = 500 * 1024 * 1024 * 1024


ORIGIN_CANONICAL = {
    "astarion": "Astarion",
    "gale": "Gale",
    "karlach": "Karlach",
    "lae'zel": "Lae'zel",
    "laezel": "Lae'zel",
    "shadowheart": "Shadowheart",
    "wyll": "Wyll",
}


APPROVAL_RE = re.compile(r"\[approval\]\s*(.+)")
CONTEXT_HEADER_RE = re.compile(r"^Context:\s*$", re.IGNORECASE)
SYNOPSIS_RE = re.compile(r"^Synopsis:\s*(.*)$", re.IGNORECASE)


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def extract_context_from_file(lines: List[str]) -> str:
    in_context = False
    context_lines: List[str] = []
    for line in lines:
        if not in_context:
            if CONTEXT_HEADER_RE.match(line):
                in_context = True
            continue
        # Stop when we hit an empty line followed by a Path header or a Path header directly
        if line.strip().startswith("Path "):
            break
        context_lines.append(line)

    context = "\n".join([l for l in context_lines if l.strip() != ""]).strip()
    if context:
        return context

    # Fallback: some files may not have explicit Context, try Synopsis
    for line in lines:
        m = SYNOPSIS_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def to_context_json_path(approval_file_path: str) -> Optional[str]:
    # Build relative path under approval-paths
    try:
        rel = os.path.relpath(approval_file_path, APPROVAL_DIR)
    except Exception:
        return None
    # Remove suffix _approval_paths.txt and change extension to .json
    base = os.path.splitext(rel)[0]
    if base.endswith("_approval_paths"):
        base = base[: -len("_approval_paths")]
    candidate_primary = os.path.join(QA_CONTEXTS_DIR_PRIMARY, base + ".json")
    if os.path.isfile(candidate_primary):
        return candidate_primary
    candidate_alt = os.path.join(QA_CONTEXTS_DIR_ALT, base + ".json")
    if os.path.isfile(candidate_alt):
        return candidate_alt
    # Even if file is missing, return primary candidate path for stable referencing
    return candidate_primary


def compute_context_relpath(approval_file_path: str) -> str:
    ctx_abs = to_context_json_path(approval_file_path)
    if not ctx_abs:
        return ""
    try:
        return os.path.relpath(ctx_abs, WORKSPACE_ROOT)
    except Exception:
        return ctx_abs


def parse_approval_payload(payload: str) -> Dict[str, int]:
    # payload example: "Gale 1, Shadowheart 1, Wyll 1, Karlach 1"
    label: Dict[str, int] = {}
    # split by comma
    parts = [p.strip() for p in payload.split(",") if p.strip()]
    for part in parts:
        # Names can contain spaces or apostrophes; number at end
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
    # Normalize whitespace and <br> tags to newlines
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    # Collapse multiple blank lines
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Remove trailing extra whitespace, keep single blanks where intentional
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def extract_paths(lines: List[str]) -> List[List[str]]:
    # Split the file into path blocks starting with "Path X:"
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


def build_conversation_and_label(path_lines: List[str]) -> Optional[Tuple[str, Dict[str, int]]]:
    # We need to capture conversation up to the LAST approval in the path.
    # Also, there can be other annotations like [context] or [description].
    # We'll scan while tracking approval occurrences and their positions.
    approval_positions: List[Tuple[int, Dict[str, int]]] = []
    processed_lines: List[str] = []

    for idx, raw in enumerate(path_lines):
        line = raw
        # Skip pure check-result metadata lines like ": True" / ": False"
        if ": True" in line or ": False" in line:
            continue
        
        text_part = line
        labels: Optional[Dict[str, int]] = None

        # Join a line that begins with [description] NodeContext: to the previous line
        # so that the node context annotation sits inline with the prior utterance.
        # This triggers only when there is a previous line to join against.
        stripped_for_nodecontext = text_part.lstrip()
        if stripped_for_nodecontext.lower().startswith("[description] NodeContext:"):
            if processed_lines:
                # Append the node context inline to the last processed line
                processed_lines[-1] = f"{processed_lines[-1].rstrip()} {stripped_for_nodecontext}"
                if labels:
                    approval_positions.append((len(processed_lines) - 1, labels))
                continue

        if "||" in line:
            left, right = line.split("||", 1)
            text_part = left.rstrip()
            appr_match = APPROVAL_RE.search(right)
            if appr_match:
                labels = parse_approval_payload(appr_match.group(1))
        else:
            # Look for approval even without explicit separator
            appr_match_inline = APPROVAL_RE.search(line)
            if appr_match_inline:
                text_part = line[: appr_match_inline.start()].rstrip()
                labels = parse_approval_payload(appr_match_inline.group(1))

        # If the current line has the same speaker as the previous line, merge them.
        # Keep the speaker prefix only on the first line and append the latter content inline.
        if processed_lines:
            prev_line = processed_lines[-1]
            prev_speaker_match = re.match(r"^([^:\[\n]+):\s*(.*)$", prev_line)
            curr_speaker_match = re.match(r"^([^:\[\n]+):\s*(.*)$", text_part)
            if prev_speaker_match and curr_speaker_match:
                prev_speaker = prev_speaker_match.group(1).strip()
                curr_speaker = curr_speaker_match.group(1).strip()
                if prev_speaker == curr_speaker:
                    # Merge by appending only the content (without the repeated speaker label)
                    merged_lead = prev_line.rstrip()
                    merged_tail = curr_speaker_match.group(2).lstrip()
                    if merged_lead and not merged_lead.endswith((" ", "\t")):
                        merged_lead += " "
                    processed_lines[-1] = merged_lead + merged_tail
                    if labels:
                        approval_positions.append((len(processed_lines) - 1, labels))
                    continue

        processed_lines.append(text_part)

        if labels:
            approval_positions.append((len(processed_lines) - 1, labels))

    if not approval_positions:
        return None

    # last approval wins
    last_index, last_labels = approval_positions[-1]

    # Construct conversation up to and including that line (but without any approval annotation)
    convo_lines = processed_lines[: last_index + 1]

    # Clean trailing/leading empties
    convo = "\n".join([ln for ln in convo_lines if ln is not None])
    convo = normalize_convo_text(convo)

    return convo, last_labels


def main() -> None:
    # To deduplicate within this run, use a set keyed by (context_rel_path, conversation)
    seen_keys = set()
    num_written = 0
    used_ids = set()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        # If file already exceeds limit, terminate immediately
        try:
            current_size = os.fstat(out.fileno()).st_size
            if current_size > MAX_OUTPUT_SIZE_BYTES:
                print(f"Output file exceeded 500GB at {OUTPUT_PATH}. Terminating.")
                try:
                    out.flush()
                    os.fsync(out.fileno())
                except Exception:
                    pass
                os.kill(os.getpid(), signal.SIGKILL)
        except Exception:
            pass

        for root, dirs, files in os.walk(APPROVAL_DIR):
            # Shuffle traversal order for directories and files to increase variety
            random.shuffle(dirs)
            random.shuffle(files)

            files = files[:100]
            for fname in tqdm(files):
                if not fname.endswith("_approval_paths.txt"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    lines = read_text(fpath)
                except Exception:
                    continue
                context_rel = compute_context_relpath(fpath)
                paths = extract_paths(lines)
                # Collect this file's samples to write in a single batch
                file_batch_lines: List[str] = []
                for path_lines in paths:
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
                    # Generate unique id (64-hex chars) with collision handling
                    base_id = hashlib.sha256(conversation.encode("utf-8")).hexdigest()
                    candidate_id = base_id
                    counter = 1
                    while candidate_id in used_ids:
                        candidate_id = hashlib.sha256((conversation + f"#{counter}").encode("utf-8")).hexdigest()
                        counter += 1
                    used_ids.add(candidate_id)

                    sample = {
                        "id": candidate_id,
                        "context": context_rel,
                        "conversation": conversation,
                        "label": label,
                    }
                    file_batch_lines.append(json.dumps(sample, ensure_ascii=False) + "\n")

                # Write this file's batch, then perform a single size check
                if file_batch_lines:
                    out.writelines(file_batch_lines)
                    out.flush()
                    num_written += len(file_batch_lines)
                    try:
                        size_bytes = os.fstat(out.fileno()).st_size
                    except Exception:
                        size_bytes = os.path.getsize(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else 0
                    print(f"File {fpath} wrote {len(file_batch_lines)} samples, total size: {size_bytes/1024**3} GB.")
                    if size_bytes > MAX_OUTPUT_SIZE_BYTES:
                        print(f"Output file exceeded 500GB at {OUTPUT_PATH}. Terminating.")
                        try:
                            out.flush()
                            os.fsync(out.fileno())
                        except Exception:
                            pass
                        os.kill(os.getpid(), signal.SIGKILL)

    print(f"Wrote {num_written} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


