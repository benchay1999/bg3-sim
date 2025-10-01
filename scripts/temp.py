import json


def merge_jsonl_files(input_file_1: str, input_file_2: str, output_file: str) -> None:
    """Merge two JSONL files into one output file, one JSON object per line."""
    with open(output_file, "w") as out_f:
        for path in (input_file_1, input_file_2):
            with open(path, "r") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    input_1 = "approval-dataset/approval_dataset.jsonl"
    input_2 = "result-dataset/approval_dataset.jsonl"
    output = "approval-dataset/approval_dataset_combined.jsonl"
    merge_jsonl_files(input_1, input_2, output)


if __name__ == "__main__":
    main()