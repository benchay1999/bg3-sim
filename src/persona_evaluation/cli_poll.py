#!/usr/bin/env python3
"""
Interactive CLI poll for BG3 approval questions with resume and balanced sampling.

Features:
- Loads items from a JSONL file containing: id, context (path to JSON), conversation,
  character, ground_truth_category (optional if we can derive from label).
- Resolves each item's context path and extracts the "context" string from the JSON.
- Builds the question as:
    Context:\n<context>\n\n\nConversation:\n<conversation>
- Presents multiple-choice options: (1) Highly approve (2) Approve (3) Neutral
  (4) Disapprove (5) Highly disapprove.
- Balanced sub-sampling: choose equal number of samples per class among
  {Highly disapprove, Disapprove, Approve, Highly approve}. Neutral is excluded
  from balanced sampling.
- Resume support: if killed, re-run with the same --output path to pick up
  where you left off. A .state.json file tracks the selected ids and current index.
- Saves answers to JSONL: {id, question, ground_truth_category, user_answer}.
- Computes user performance metrics similar to run_llm_approval_inference.py.

Usage examples:
  # Full run (no sub-sampling; all four non-neutral categories included)
  python src/cli_poll.py \
      --input /home/wschay/bg3-sim/test/1002_gpt-4o-mini_astarion_llm_approvals.jsonl \
      --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl \
      --base-dir /home/wschay/bg3-sim

  # Balanced sub-sample with 20 per class (total up to 80)
  python src/cli_poll.py \
      --input /home/wschay/bg3-sim/test/1002_gpt-4o-mini_astarion_llm_approvals.jsonl \
      --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl \
      --base-dir /home/wschay/bg3-sim \
      --per_class 20 --seed 123

  # Resume a previous session (just re-run with same --output path)
  python src/cli_poll.py --input ... --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl

  # Compute stats only (without running the poll)
  python src/cli_poll.py --input ... --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl --stats-only
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_ORDER = [
    "Highly disapprove",
    "Disapprove",
    "Approve",
    "Highly approve",
]

CANONICAL_MAP = {
    "highly disapprove": "Highly disapprove",
    "disapprove": "Disapprove",
    "approve": "Approve",
    "highly approve": "Highly approve",
    "neutral": "Neutral",
}


def canonicalize_category(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    return CANONICAL_MAP.get(name.strip().lower())


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def categorize_by_ranges(value: Any) -> Optional[str]:
    v = to_float(value)
    if v is None:
        return None
    if v <= -5:
        return "Highly disapprove"
    if -5 < v <= -1:
        return "Disapprove"
    if 1 <= v < 5:
        return "Approve"
    if v >= 5:
        return "Highly approve"
    return None


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def resolve_context_path(context_path: str, base_dir: str) -> Optional[str]:
    if context_path.startswith("@"):  # allow @prefix
        context_path = context_path[1:]
    if os.path.isabs(context_path) and os.path.exists(context_path):
        return context_path
    candidate = os.path.join(base_dir, context_path)
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.abspath(context_path)
    if os.path.exists(candidate2):
        return candidate2
    return None


def extract_context_text(context_json_path: Optional[str]) -> str:
    if not context_json_path:
        return ""
    try:
        data = read_json(context_json_path)
        if isinstance(data, dict) and isinstance(data.get("context"), str):
            return data["context"].strip()
    except Exception:
        return ""
    return ""


def build_question(context_text: str, conversation_text: str) -> str:
    return f"Context:\n{context_text}\n\n\nConversation:\n{conversation_text}"


def derive_ground_truth_category(row: Dict[str, Any], default_character: str) -> Optional[str]:
    gt = row.get("ground_truth_category")
    canonical = canonicalize_category(gt)
    if canonical:
        return canonical
    # Derive from numeric label if present
    label = row.get("label")
    character_name = row.get("character") if isinstance(row.get("character"), str) else default_character
    if isinstance(label, dict) and character_name in label:
        return categorize_by_ranges(label.get(character_name))
    return None


def compute_selection(
    rows: List[Dict[str, Any]],
    per_class: int,
    rng: random.Random,
    include_categories: List[str],
) -> List[str]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in include_categories}
    for r in rows:
        cat = derive_ground_truth_category(r, default_character="Astarion")
        if cat in by_cat:
            by_cat[cat].append(r)

    for c in by_cat:
        rng.shuffle(by_cat[c])

    selected_ids: List[str] = []
    if per_class and per_class > 0:
        for c in include_categories:
            bucket = by_cat.get(c, [])
            take = bucket[: min(per_class, len(bucket))]
            selected_ids.extend([t.get("id") for t in take if isinstance(t.get("id"), str)])
    else:
        # Take all across categories
        for c in include_categories:
            bucket = by_cat.get(c, [])
            selected_ids.extend([t.get("id") for t in bucket if isinstance(t.get("id"), str)])

    # Shuffle overall order to mix categories
    rng.shuffle(selected_ids)
    return selected_ids


def load_answers(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    answers: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("id")
                if isinstance(sid, str):
                    answers[sid] = obj
            except Exception:
                continue
    return answers


def save_answer_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def save_state(path: str, state: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def load_state(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def map_choice_to_category(choice: str) -> Optional[str]:
    mapping = {
        "1": "Highly disapprove",
        "2": "Disapprove",
        "3": "Neutral",
        "4": "Approve",
        "5": "Highly approve",
    }
    return mapping.get(choice.strip())


def compute_metrics(answers: List[Dict[str, Any]], metrics_dir: str) -> None:
    # try to import sklearn & matplotlib; otherwise print basic accuracy only
    y_true: List[str] = []
    y_pred: List[str] = []

    for a in answers:
        gt = canonicalize_category(a.get("ground_truth_category"))
        ua = canonicalize_category(a.get("user_answer"))
        if gt and ua:
            y_true.append(gt)
            y_pred.append(ua)

    if not y_true:
        print("No comparable answers; skipping metrics.")
        return

    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        # basic accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        acc = correct / len(y_true)
        print(f"Accuracy: {acc:.4f} ({correct}/{len(y_true)})")
        return

    labels_order = ["Highly disapprove", "Disapprove", "Approve", "Highly approve"]
    target_names = list(labels_order)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, zero_division=0
    )

    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        "accuracy": acc,
        "per_class": {
            labels_order[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels_order))
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels_order, target_names=target_names, zero_division=0
        ),
    }
    metrics_path = os.path.join(metrics_dir, "user_poll_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set(xticks=range(len(labels_order)), yticks=range(len(labels_order)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = os.path.join(metrics_dir, "user_poll_confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    # --- Binary metrics (positive = Approve/Highly approve; negative = Disapprove/Highly disapprove) ---
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    for t, p in zip(y_true, y_pred):
        true_bin: Optional[int] = None
        pred_bin: Optional[int] = None
        if t in ("Approve", "Highly approve"):
            true_bin = 1
        elif t in ("Disapprove", "Highly disapprove"):
            true_bin = 0
        if p in ("Approve", "Highly approve"):
            pred_bin = 1
        elif p in ("Disapprove", "Highly disapprove"):
            pred_bin = 0
        # Skip cases where either side is Neutral or missing
        if true_bin is not None and pred_bin is not None:
            y_true_bin.append(true_bin)
            y_pred_bin.append(pred_bin)

    if len(y_true_bin) > 0:
        acc_bin = float(accuracy_score(y_true_bin, y_pred_bin))
        prec_bin, rec_bin, f1_bin, sup_bin = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, labels=[0, 1], zero_division=0
        )
        metrics["binary"] = {
            "accuracy": acc_bin,
            "per_class": {
                "negative": {
                    "precision": float(prec_bin[0]),
                    "recall": float(rec_bin[0]),
                    "f1": float(f1_bin[0]),
                    "support": int(sup_bin[0]),
                },
                "positive": {
                    "precision": float(prec_bin[1]),
                    "recall": float(rec_bin[1]),
                    "f1": float(f1_bin[1]),
                    "support": int(sup_bin[1]),
                },
            },
            "classification_report": classification_report(
                y_true_bin, y_pred_bin, labels=[0, 1], target_names=["negative", "positive"], zero_division=0
            ),
        }
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        print(f"Updated metrics with binary results at {metrics_path}")

        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1)
        im2 = ax2.imshow(cm_bin, interpolation='nearest', cmap=plt.cm.Blues)
        fig2.colorbar(im2)
        ax2.set(xticks=[0, 1], yticks=[0, 1])
        ax2.set_xticklabels(["negative", "positive"], rotation=45, ha="right")
        ax2.set_yticklabels(["negative", "positive"]) 
        ax2.set_ylabel('True label')
        ax2.set_xlabel('Predicted label')
        thresh2 = cm_bin.max() / 2.0 if cm_bin.size > 0 else 0
        for i in range(cm_bin.shape[0]):
            for j in range(cm_bin.shape[1]):
                ax2.text(j, i, format(cm_bin[i, j], 'd'), ha="center", va="center",
                        color="white" if cm_bin[i, j] > thresh2 else "black")
        fig2.tight_layout()
        cm_bin_path = os.path.join(metrics_dir, "user_poll_confusion_matrix_binary.png")
        fig2.savefig(cm_bin_path, dpi=200)
        plt.close(fig2)
        print(f"Saved binary confusion matrix to {cm_bin_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive CLI poll with resume and balanced sampling.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to write answers JSONL (resume if exists).")
    parser.add_argument("--base-dir", default=os.getcwd(), help="Base dir for resolving context JSON paths.")
    parser.add_argument("--per_class", type=int, default=0, help="Samples per class for balanced sampling (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection.")
    parser.add_argument("--state", default=None, help="Optional path for state file (.state.json). Defaults to output with suffix.")
    parser.add_argument("--stats-only", action="store_true", help="Only compute stats from existing output and exit.")
    parser.add_argument("--metrics-dir", default=None, help="Directory to save metrics and plots (default: output's directory).")
    args = parser.parse_args()

    if args.stats_only:
        answers_map = load_answers(args.output)
        answers_list = list(answers_map.values())
        metrics_dir = args.metrics_dir or os.path.dirname(args.output) or "."
        compute_metrics(answers_list, metrics_dir)
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    state_path = args.state or (os.path.splitext(args.output)[0] + ".state.json")

    rng = random.Random(args.seed)

    # Load input rows
    rows = read_jsonl(args.input)
    # Pre-compute a map by id for fast lookup
    id_to_row: Dict[str, Dict[str, Any]] = {}
    usable_rows: List[Dict[str, Any]] = []
    for r in rows:
        sid = r.get("id")
        if isinstance(sid, str):
            # Filter to only rows with a usable ground-truth among four classes
            cat = derive_ground_truth_category(r, default_character="Astarion")
            if cat in CANONICAL_ORDER:
                id_to_row[sid] = r
                usable_rows.append(r)

    # Load existing answers (for dedupe and to compute current position if state missing)
    answers_map = load_answers(args.output)
    answered_ids = set(answers_map.keys())

    # Load or create state
    state = load_state(state_path)
    input_abs = os.path.abspath(args.input)
    output_abs = os.path.abspath(args.output)
    base_abs = os.path.abspath(args.base_dir)

    needs_reselect = True
    if state and isinstance(state.get("selected_ids"), list):
        # Validate that state matches current configuration; if not, reselect
        same_input = state.get("input_path") == input_abs
        same_output = state.get("output_path") == output_abs
        same_base = state.get("base_dir") == base_abs
        same_per_class = int(state.get("per_class", -1)) == int(args.per_class)
        same_seed = int(state.get("random_seed", -1)) == int(args.seed)
        if same_input and same_output and same_base and same_per_class and same_seed:
            needs_reselect = False

    if not needs_reselect:
        selected_ids: List[str] = [s for s in state["selected_ids"] if isinstance(s, str) and s in id_to_row]
        current_index: int = int(state.get("current_index", 0))
        while current_index < len(selected_ids) and selected_ids[current_index] in answered_ids:
            current_index += 1
        state["current_index"] = current_index
        # Persist any index advancement
        save_state(state_path, state)
    else:
        selected_ids = compute_selection(
            usable_rows,
            per_class=args.per_class,
            rng=rng,
            include_categories=list(CANONICAL_ORDER),
        )
        current_index = 0
        while current_index < len(selected_ids) and selected_ids[current_index] in answered_ids:
            current_index += 1
        state = {
            "input_path": input_abs,
            "output_path": output_abs,
            "base_dir": base_abs,
            "per_class": args.per_class,
            "random_seed": args.seed,
            "selected_ids": selected_ids,
            "current_index": current_index,
        }
        save_state(state_path, state)

    total = len(selected_ids)
    if total == 0:
        print("No samples available for the selected configuration.")
        return

    print(f"Starting poll: {total} questions (index {state['current_index']}..{total-1}).")
    print("Answer with 1) Highly disapprove  2) Disapprove  3) Neutral  4) Approve  5) Highly approve")
    print("Press 'q' then Enter to quit at any time (progress is saved).\n")

    try:
        idx = state["current_index"]
        while idx < total:
            sid = selected_ids[idx]
            row = id_to_row.get(sid)
            if not row:
                idx += 1
                continue

            # Resolve context and build question
            ctx_path_raw = row.get("context", "")
            ctx_path_abs = resolve_context_path(str(ctx_path_raw), args.base_dir)
            context_text = extract_context_text(ctx_path_abs)
            conversation_text = row.get("conversation", "")
            question_text = build_question(context_text, conversation_text)

            gt_category = derive_ground_truth_category(row, default_character="Astarion") or ""

            print(f"[{idx+1}/{total}] id={sid}")
            print(question_text)
            print("")
            print("Your choice [1-5] (q to quit)\n\n(1): Highly disapprove (2) Disapprove (3) Neutral (4) Approve (5) Highly approve\n\n", end="", flush=True)
            choice = input().strip().lower()
            if choice == "q":
                print("Quitting. Progress saved.")
                break
            user_category = map_choice_to_category(choice)
            if not user_category:
                print("Invalid choice. Please enter a number 1-5, or 'q' to quit.")
                continue

            answer_obj = {
                "id": sid,
                "question": question_text,
                "ground_truth_category": gt_category,
                "user_answer": user_category,
            }
            save_answer_line(args.output, answer_obj)
            answers_map[sid] = answer_obj

            idx += 1
            state["current_index"] = idx
            save_state(state_path, state)
            print("")

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")

    # Final stats
    answers_list = list(answers_map.values())
    metrics_dir = args.metrics_dir or os.path.dirname(args.output) or "."
    compute_metrics(answers_list, metrics_dir)


if __name__ == "__main__":
    main()


