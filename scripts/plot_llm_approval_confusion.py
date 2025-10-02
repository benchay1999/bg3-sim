#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


NON_NEUTRAL_LABELS = [
    "Highly disapprove",
    "Disapprove",
    "Approve",
    "Highly approve",
]


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


def category_to_binary(value: Optional[str]) -> Optional[int]:
    if value in ("Approve", "Highly approve"):
        return 1
    if value in ("Disapprove", "Highly disapprove"):
        return 0
    return None


def collect_arrays(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[int], List[int], int, str, str]:
    y_true: List[str] = []
    y_pred: List[str] = []
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    predicted_neutral_count: int = 0

    character_name: str = "Unknown"
    model_name: str = "model"

    for row in rows:
        if "character" in row and isinstance(row["character"], str):
            character_name = row["character"]
        if "model" in row and isinstance(row["model"], str):
            model_name = row["model"]

        gt: Optional[str] = row.get("ground_truth_category")
        pred: Optional[str] = row.get("predicted_approval")

        if pred == "Neutral":
            predicted_neutral_count += 1

        if gt in NON_NEUTRAL_LABELS and pred in NON_NEUTRAL_LABELS:
            y_true.append(gt) 
            y_pred.append(pred) 

        gt_bin = category_to_binary(gt)
        pred_bin = category_to_binary(pred)
        if gt_bin is not None and pred_bin is not None:
            y_true_bin.append(gt_bin)
            y_pred_bin.append(pred_bin)

    return y_true, y_pred, y_true_bin, y_pred_bin, predicted_neutral_count, character_name, model_name


def plot_multiclass(y_true: List[str], y_pred: List[str], predicted_neutral_count: int, character: str, model: str, out_dir: str) -> Optional[str]:
    if not y_true:
        return None

    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels_order = list(NON_NEUTRAL_LABELS)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.05, 0.35], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[0, 2])

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, cax=cax)
    ax.set(xticks=range(len(labels_order)), yticks=range(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=45, ha="right")
    ax.set_yticklabels(labels_order)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.suptitle(f"{character} Approval Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    lax.axis('off')
    lax.text(0.0, 0.5, f"Predicted Neutral: {predicted_neutral_count}", ha="left", va="center",
             fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="lightgray"))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    character_slug = character.lower().replace(" ", "_")
    model_slug = model.split('/')[-1].lower()
    out_path = os.path.join(out_dir, f"{model_slug}_{character_slug}_llm_confusion_matrix.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_binary(y_true_bin: List[int], y_pred_bin: List[int], predicted_neutral_count: int, character: str, model: str, out_dir: str) -> Optional[str]:
    if not y_true_bin:
        return None

    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    fig = plt.figure(figsize=(7, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.05, 0.35], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[0, 2])

    im = ax.imshow(cm_bin, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, cax=cax)
    ax.set(xticks=[0, 1], yticks=[0, 1])
    ax.set_xticklabels(["negative", "positive"], rotation=45, ha="right")
    ax.set_yticklabels(["negative", "positive"]) 
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.suptitle(f"{character} Approval Binary Confusion Matrix")

    thresh2 = cm_bin.max() / 2.0 if cm_bin.size > 0 else 0
    for i in range(cm_bin.shape[0]):
        for j in range(cm_bin.shape[1]):
            ax.text(j, i, format(cm_bin[i, j], 'd'), ha="center", va="center",
                    color="white" if cm_bin[i, j] > thresh2 else "black")

    lax.axis('off')
    lax.text(0.0, 0.5, f"Predicted Neutral: {predicted_neutral_count}", ha="left", va="center",
             fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="lightgray"))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    character_slug = character.lower().replace(" ", "_")
    model_slug = model.split('/')[-1].lower()
    out_path = os.path.join(out_dir, f"{model_slug}_{character_slug}_llm_confusion_matrix_binary.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrices from LLM approval inference output JSONL.")
    parser.add_argument("--input", required=True, help="Path to inference output JSONL file.")
    parser.add_argument("--output_dir", default="test", help="Directory to save plots.")
    parser.add_argument("--character", default=None, help="Optional character name to label plots; defaults to value in file.")
    parser.add_argument("--model", default=None, help="Optional model name to label plots; defaults to value in file.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        print("No rows found in input JSONL; nothing to plot.")
        return

    y_true, y_pred, y_true_bin, y_pred_bin, predicted_neutral_count, character_from_file, model_from_file = collect_arrays(rows)

    character = args.character if isinstance(args.character, str) and len(args.character) > 0 else character_from_file
    model = args.model if isinstance(args.model, str) and len(args.model) > 0 else model_from_file

    multi_path = plot_multiclass(y_true, y_pred, predicted_neutral_count, character, model, args.output_dir)
    if multi_path:
        print(f"Saved multiclass confusion matrix to {multi_path}")
    else:
        print("Skipping multiclass plot (no eligible samples).")

    bin_path = plot_binary(y_true_bin, y_pred_bin, predicted_neutral_count, character, model, args.output_dir)
    if bin_path:
        print(f"Saved binary confusion matrix to {bin_path}")
    else:
        print("Skipping binary plot (no eligible samples).")


if __name__ == "__main__":
    main()


