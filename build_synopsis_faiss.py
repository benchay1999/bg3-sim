import argparse
import json
import os
import sys

from dialog_simulator import DialogSimulator


def find_seed_json(cluster_index_file: str) -> str:
    try:
        with open(cluster_index_file, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
    except Exception as e:
        print(f"Failed to read cluster index '{cluster_index_file}': {e}")
        return ''

    # Find the first existing JSON path in any cluster to seed DialogSimulator
    for _, paths in clusters.items():
        for p in paths:
            if os.path.isfile(p):
                return p
    return ''


def main():
    parser = argparse.ArgumentParser(description="Build a global FAISS index over all session synopses.")
    parser.add_argument("--cluster-index", default="goals_to_json_paths.json", help="Path to goals_to_json_paths.json")
    parser.add_argument("--save-dir", default="vector_db_synopses", help="Directory to write index.faiss and meta.json")
    parser.add_argument("--model", default=None, help="Sentence-Transformers model id (e.g., all-MiniLM-L6-v2). Default: DialogSimulator's")
    args = parser.parse_args()

    seed_json = find_seed_json(args.cluster_index)
    if not seed_json:
        print("Could not locate any JSON file from the cluster index to initialize the simulator.")
        sys.exit(1)

    print(f"Seeding simulator with: {seed_json}")
    sim = DialogSimulator(seed_json)

    ok = sim.build_global_synopsis_faiss(
        cluster_index_file=args.cluster_index,
        model_name=args.model,
        save_dir=args.save_dir,
    )
    if not ok:
        print("Failed to build global synopsis FAISS index.")
        sys.exit(2)

    print(f"Global synopsis FAISS index written to: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()


