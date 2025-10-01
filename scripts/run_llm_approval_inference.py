#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Workspace root for resolving relative context paths saved in the dataset
WORKSPACE_ROOT = "/home/wschay/bg3sim"


def load_persona_template(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def build_prompt_from_template(template_lines: List[str], context_text: str, conversation_text: str) -> str:
    # Keep everything up to and including the first line that starts with "# Conversation"
    out_lines: List[str] = []
    inserted = False
    for line in template_lines:
        out_lines.append(line)
        if line.strip().startswith("# Conversation"):
            inserted = True
            break

    if not inserted:
        # If the template has no explicit Conversation anchor, just append at the end
        out_lines.append("# Conversation")

    # Insert per-sample block
    out_lines.append(f"context: {context_text}".rstrip())
    out_lines.append("")
    out_lines.extend(conversation_text.splitlines())

    return "\n".join(out_lines)


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
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_model_response(text: str) -> Dict[str, Optional[Any]]:
    """Parse model output. Expect JSON with player_reason and player_approval.
    If not valid JSON, try to extract approval category via keyword search.
    """
    approval = None
    reason = None

    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if isinstance(data.get("player_approval"), str):
                approval = data.get("player_approval").strip()
            if isinstance(data.get("player_reason"), str):
                reason = data.get("player_reason").strip()
    except Exception:
        pass

    # Fallback: keyword search in raw text
    if not approval and isinstance(text, str):
        lower = text.lower()
        # Order matters to avoid substring clashes
        if "highly disapprove" in lower:
            approval = "Highly disapprove"
        elif "highly approve" in lower:
            approval = "Highly approve"
        elif "disapprove" in lower:
            approval = "Disapprove"
        elif "approve" in lower:
            approval = "Approve"
        elif "neutral" in lower:
            approval = "Neutral"

    return {
        "player_reason": reason,
        "player_approval": approval,
    }


# Canonical category names for modeling approvals (case-sensitive canonical forms)
CATEGORY_CANONICAL = {
    "highly disapprove": "Highly disapprove",
    "disapprove": "Disapprove",
    "approve": "Approve",
    "highly approve": "Highly approve",
    "neutral": "Neutral",
}


def canonicalize_category(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    return CATEGORY_CANONICAL.get(name.strip().lower())


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def categorize_by_ranges(value: Any) -> Optional[str]:
    """Map numeric approval to category based on ranges:
    - Highly disapprove: score <= -5
    - Disapprove: -5 < score <= -1
    - Approve: 1 <= score < 5
    - Highly approve: score >= 5
    Returns canonical category name or None if not in any bucket.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiteLLM approvals for BG3 samples using a persona template.")
    parser.add_argument("--input", required=True, help="Path to dataset file (JSON array or JSONL).")
    parser.add_argument("--output", required=True, help="Output JSONL path for model responses.")
    parser.add_argument("--template", default=f"{WORKSPACE_ROOT}/bg3_characters_llm_input_prompt_example.txt", help="Persona template path.")
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model name/id.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples (0 = all).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--metrics_dir", default=f"{WORKSPACE_ROOT}/test", help="Directory to save metrics and plots.")
    parser.add_argument("--character", default="Astarion", help="Character name to use for ground-truth labels (e.g., Astarion, Lae'zel, Gale, Wyll, Shadowheart).")
    parser.add_argument("--api_base", default=None, help="Optional OpenAI-compatible base URL (e.g., http://localhost:8010/v1 for vLLM).")
    parser.add_argument("--api_key", default=None, help="Optional API key for OpenAI-compatible endpoints. Use a dummy if not required.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional request timeout (seconds).")
    args = parser.parse_args()

    # Load .env for API keys if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # LiteLLM import (match style from dialog_simulator)
    try:
        from litellm import completion as litellm_completion
    except Exception:
        print("LiteLLM is not installed. Install with: pip install litellm")
        sys.exit(1)

    # Load dataset (supports JSON array or JSONL)
    samples: List[Dict[str, Any]]
    if args.input.endswith(".jsonl"):
        samples = read_jsonl(args.input)
    else:
        data = read_json(args.input)
        if isinstance(data, list):
            samples = data
        else:
            raise ValueError("Input JSON must be an array or a JSONL file.")

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    template_lines = load_persona_template(args.template)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    # Open in append mode to persist progress across interruptions
    out_f = open(args.output, "a", encoding="utf-8")

    # Multiclass categories (strings) and binary polarity metrics
    y_true: List[str] = []
    y_pred: List[str] = []
    y_true_bin: List[int] = []  # 1 = positive, 0 = negative
    y_pred_bin: List[int] = []

    sent = 0
    for sample in tqdm(samples):
        context_rel: str = sample.get("context", "")
        conversation_text: str = sample.get("conversation", "")
        label: Dict[str, Any] = sample.get("label", {})

        # Load context text from the QA JSON referenced by relative path
        ctx_abs = os.path.join(WORKSPACE_ROOT, context_rel)
        context_text = ""
        try:
            ctx_data = read_json(ctx_abs)
            if isinstance(ctx_data, dict) and isinstance(ctx_data.get("context"), str):
                context_text = ctx_data["context"].strip()
        except Exception:
            context_text = ""

        prompt_text = build_prompt_from_template(template_lines, context_text, conversation_text)

        try:
            kwargs = {}
            if args.api_base:
                kwargs["api_base"] = args.api_base
            if args.api_key:
                kwargs["api_key"] = args.api_key
            if args.timeout is not None:
                kwargs["timeout"] = args.timeout
            resp = litellm_completion(
                model=args.model,
                messages=[
                    {"role": "user", "content": prompt_text},
                ],
                max_completion_tokens=1000,
                **kwargs
            )
            content = resp.choices[0].message["content"].strip()
        except Exception as e:
            content = f"ERROR: {e}"
        parsed = parse_model_response(content)
        predicted_approval_raw = parsed.get("player_approval")
        predicted_category = canonicalize_category(predicted_approval_raw)

        character_label_value = None
        if isinstance(label, dict) and args.character in label:
            character_label_value = label.get(args.character)

        ground_truth_category = categorize_by_ranges(character_label_value)

        correct: Optional[bool] = None
        if predicted_category is not None and ground_truth_category is not None:
            correct = (predicted_category == ground_truth_category)

        # Binary correctness: positive (Approve/Highly approve) vs negative (Disapprove/Highly disapprove)
        binary_correct: Optional[bool] = None
        binary_true: Optional[int] = None
        binary_pred: Optional[int] = None
        if ground_truth_category in ("Approve", "Highly approve"):
            binary_true = 1
        elif ground_truth_category in ("Disapprove", "Highly disapprove"):
            binary_true = 0
        if predicted_category in ("Approve", "Highly approve"):
            binary_pred = 1
        elif predicted_category in ("Disapprove", "Highly disapprove"):
            binary_pred = 0
        if binary_true is not None and binary_pred is not None:
            binary_correct = (binary_true == binary_pred)

        out_obj = {
            "context": context_rel,
            "conversation": conversation_text,
            "label": label,
            "model": args.model,
            "response": content,
            "character": args.character,
            "ground_truth_category": ground_truth_category,
            "predicted_approval": predicted_category,
            "correct": correct,
            "binary_correct": binary_correct,
        }
        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        # Flush and fsync to persist each sample immediately
        out_f.flush()
        try:
            os.fsync(out_f.fileno())
        except Exception:
            pass
        sent += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

        # Collect metrics when both categories are available and among the four non-neutral classes
        if ground_truth_category in ("Highly disapprove", "Disapprove", "Approve", "Highly approve") and \
           predicted_category in ("Highly disapprove", "Disapprove", "Approve", "Highly approve"):
            y_true.append(ground_truth_category)
            y_pred.append(predicted_category)
            # Binary arrays already populated via binary_true/binary_pred
            if binary_true is not None and binary_pred is not None:
                y_true_bin.append(binary_true)
                y_pred_bin.append(binary_pred)

    out_f.close()
    print(f"Completed {sent} requests. Wrote: {args.output}")

    # Compute metrics
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        print("scikit-learn/matplotlib not available; skipping metrics and plots.")
        return

    if not y_true:
        print("No valid predictions with ground-truth labels; skipping metrics.")
        return

    labels_order = ["Highly disapprove", "Disapprove", "Approve", "Highly approve"]
    target_names = list(labels_order)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, zero_division=0
    )

    # Save metrics JSON
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
    character_slug = args.character.lower().replace(" ", "_")
    metrics_path = os.path.join(args.metrics_dir, f"{args.model}_{character_slug}_llm_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Confusion matrix plot (multiclass)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels_order)), yticks=range(len(labels_order)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(f"{args.character} Approval Confusion Matrix")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    cm_path = os.path.join(args.metrics_dir, f"{args.model}_{character_slug}_llm_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    # Binary metrics and confusion matrix if we have any binary pairs
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
        # Save updated metrics including binary
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        print(f"Updated metrics with binary results at {metrics_path}")

        # Plot binary confusion matrix
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(cm_bin, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.figure.colorbar(im2, ax=ax2)
        ax2.set(xticks=[0, 1], yticks=[0, 1])
        ax2.set_xticklabels(["negative", "positive"], rotation=45, ha="right")
        ax2.set_yticklabels(["negative", "positive"])
        ax2.set_ylabel('True label')
        ax2.set_xlabel('Predicted label')
        ax2.set_title(f"{args.character} Approval Binary Confusion Matrix")
        thresh2 = cm_bin.max() / 2.0 if cm_bin.size > 0 else 0
        for i in range(cm_bin.shape[0]):
            for j in range(cm_bin.shape[1]):
                ax2.text(j, i, format(cm_bin[i, j], 'd'), ha="center", va="center",
                        color="white" if cm_bin[i, j] > thresh2 else "black")
        plt.tight_layout()
        cm_bin_path = os.path.join(args.metrics_dir, f"{args.model}_{character_slug}_llm_confusion_matrix_binary.png")
        plt.savefig(cm_bin_path, dpi=200)
        plt.close(fig2)
        print(f"Saved binary confusion matrix to {cm_bin_path}")


if __name__ == "__main__":
    main()


