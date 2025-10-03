#!/usr/bin/env python3
import json
import os
from collections import Counter, defaultdict


DATASET_PATH = "/home/wschay/bg3sim/result-dataset/approval_dataset.jsonl"


def main() -> None:
    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        return

    global_conv_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple] = Counter()
    examples_by_conv = defaultdict(list)
    examples_by_pair = defaultdict(list)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversation", "")
            ctx = obj.get("context", "")
            global_conv_counter[conv] += 1
            pair_counter[(ctx, conv)] += 1
            if len(examples_by_conv[conv]) < 3:
                examples_by_conv[conv].append(obj)
            if len(examples_by_pair[(ctx, conv)]) < 3:
                examples_by_pair[(ctx, conv)].append(obj)

    dup_global = {k: c for k, c in global_conv_counter.items() if c > 1}
    dup_pairs = {k: c for k, c in pair_counter.items() if c > 1}

    print("Global duplicate conversation strings:")
    print(f"  unique with duplicates: {len(dup_global)}")
    if dup_global:
        print("  top 10:")
        for conv, c in sorted(dup_global.items(), key=lambda x: -x[1])[:10]:
            snippet = conv[:120].replace("\n", " ")
            print(f"    count={c} | snippet={snippet}...")

    print("\nDuplicate (context, conversation) pairs:")
    print(f"  unique with duplicates: {len(dup_pairs)}")
    if dup_pairs:
        print("  top 10:")
        for (ctx, conv), c in sorted(dup_pairs.items(), key=lambda x: -x[1])[:10]:
            snippet = conv[:120].replace("\n", " ")
            print(f"    count={c} | context={ctx} | snippet={snippet}...")


if __name__ == "__main__":
    main()


