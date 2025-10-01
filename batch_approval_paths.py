import sys
import json
from pathlib import Path
import time

from dialog_simulator import DialogSimulator


def main():
    """Batch-run approval-only simulations over all dialog JSON files in 'output/'.

    For each JSON file found, runs approval-only path discovery and writes:
      - A human-readable TXT of the approval-only paths
      - A structured JSON of traversal data for approval-only paths

    Outputs are written under a mirrored directory tree inside 'approval-paths/'.
    """
    output_root = Path('output')
    dest_root = Path('approval-paths')

    if not output_root.exists():
        print(f"Input directory not found: {output_root.resolve()}")
        sys.exit(1)

    dest_root.mkdir(parents=True, exist_ok=True)

    json_files = sorted(output_root.rglob('*.json'))
    if not json_files:
        print("No JSON files found under 'output/'. Nothing to process.")
        return

    manifest = {
        'total_files': len(json_files),
        'processed': [],
        'errors': []
    }
    total_approval_paths = 0
    files_with_approvals = 0
    files_without_approvals = 0
    per_file_counts = []  # (source_path, num_paths)

    for json_path in json_files:
        try:
            rel_path = json_path.relative_to(output_root)
        except ValueError:
            # Fallback if the file is not actually under output_root for some reason
            rel_path = json_path.name

        print(f"\n=== Processing: {json_path} ===")
        simulator = DialogSimulator(str(json_path))
        if "SHA_Merregon_000.json" in str(json_path) or "LOW_StormshoreTabernacle_TyrShrine.json" in str(json_path) or "CAMP_Courier_Dog.json" in str(json_path):
            continue
        # Run approval-only simulation without exporting (we'll control destinations)
        approval_paths, _, _, _ = simulator.simulate_approval_paths(
            max_depth=25,
            print_paths=False,
            test_mode=True,
            export_txt=False,
            export_json=False,
            export_dict=False,
            verbose=False,
            time_limit_seconds=1200
        )

        if not approval_paths:
            print("No approval paths found. Skipping outputs for this file.")
            files_without_approvals += 1
            continue

        files_with_approvals += 1
        total_approval_paths += len(approval_paths)
        per_file_counts.append((str(json_path), len(approval_paths)))

        dest_dir = dest_root / rel_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        base_name = json_path.stem
        txt_out = dest_dir / f"{base_name}_approval_paths.txt"
        json_out = dest_dir / f"{base_name}_approval_traversals.json"

        # Save human-readable TXT
        simulator.export_paths_to_txt(approval_paths, output_file=str(txt_out))

        # Save structured JSON traversals
        traversals = []
        for idx, path in enumerate(approval_paths, 1):
            start_time = time.perf_counter()
            single_traversal = simulator.create_traversal_data([path])[0]
            elapsed = time.perf_counter() - start_time
            traversals.append(single_traversal)
            print(f"Traversal {idx} time: {elapsed:.4f}s (nodes: {len(single_traversal)})")

        simulator.export_traversals_to_json(traversals, output_file=str(json_out))

        manifest['processed'].append({
            'source': str(json_path),
            'txt': str(txt_out),
            'json': str(json_out),
            'num_paths': len(approval_paths)
        })

    # Write a manifest for convenience
    manifest_path = dest_root / '_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote manifest: {manifest_path}")

    # Write summary statistics
    stats_path = dest_root / '_stats.txt'
    avg_paths = (total_approval_paths / files_with_approvals) if files_with_approvals else 0
    top_n = 10
    top_files = sorted(per_file_counts, key=lambda x: x[1], reverse=True)[:top_n]
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Approval Paths Summary\n")
        f.write("=======================\n\n")
        f.write(f"Total JSON files scanned: {len(json_files)}\n")
        f.write(f"Files with approval paths: {files_with_approvals}\n")
        f.write(f"Files without approval paths: {files_without_approvals}\n")
        f.write(f"Total approval paths: {total_approval_paths}\n")
        f.write(f"Average approval paths per file (with approvals): {avg_paths:.2f}\n")
        if per_file_counts:
            f.write("\nTop files by approval path count:\n")
            for i, (src, count) in enumerate(top_files, 1):
                f.write(f"{i}. {src} -> {count}\n")
    print(f"Wrote statistics: {stats_path}")


if __name__ == '__main__':
    main()


