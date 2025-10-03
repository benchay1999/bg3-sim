import sys
import json
from pathlib import Path
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from dialog_simulator import DialogSimulator


def _process_single_file(json_path_str, output_root_str, dest_root_str):
    """Process one JSON file and return a result dict for aggregation."""
    try:
        json_path = Path(json_path_str)
        output_root = Path(output_root_str)
        dest_root = Path(dest_root_str)

        # Determine relative path under output/ for mirrored destination
        try:
            rel_path = json_path.relative_to(output_root)
        except ValueError:
            # Fallback if the file is not actually under output_root
            rel_path = Path(json_path.name)

        dest_dir = dest_root / rel_path.parent
        base_name = json_path.stem
        txt_out = dest_dir / f"{base_name}_approval_paths.txt"
        json_out = dest_dir / f"{base_name}_approval_traversals.json"

        # Skip if outputs already exist
        if txt_out.exists() and json_out.exists():
            return {'skipped_existing': True, 'source': str(json_path)}

        # Per-file lock to avoid duplicate work within/between processes
        lock_path = dest_dir / f"{base_name}.lock"
        dest_dir.mkdir(parents=True, exist_ok=True)
        lock_acquired = False
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
        except FileExistsError:
            return {'skipped_in_progress': True, 'source': str(json_path)}

        # File-specific skip rules
        s = str(json_path)
        if ("SHA_Merregon_000.json" in s) or ("LOW_StormshoreTabernacle_TyrShrine.json" in s) or ("CAMP_Courier_Dog.json" in s):
            # Clean up lock if we created it
            if lock_acquired and lock_path.exists():
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
            return {'skipped_rule': True, 'source': str(json_path)}

        simulator = DialogSimulator(str(json_path))
        approval_paths, _, _, _ = simulator.simulate_approval_paths(
            max_depth=50,
            print_paths=False,
            test_mode=True,
            export_txt=False,
            export_json=False,
            export_dict=False,
            verbose=False,
            time_limit_seconds=30
        )

        # Filter out paths that exceeded time limit or max depth
        filtered_paths = [p for p in approval_paths if ("TIMEOUT_REACHED" not in p and "MAX_DEPTH_REACHED" not in p)]

        if not filtered_paths:
            return {'no_approvals': True, 'source': str(json_path)}

        # Save human-readable TXT
        simulator.export_paths_to_txt(filtered_paths, output_file=str(txt_out))

        # Save structured JSON traversals
        traversals = []
        for idx, path in enumerate(filtered_paths, 1):
            start_time = time.perf_counter()
            single_traversal = simulator.create_traversal_data([path])[0]
            elapsed = time.perf_counter() - start_time
            traversals.append(single_traversal)
            print(f"Traversal {idx} time: {elapsed:.4f}s (nodes: {len(single_traversal)})")

        simulator.export_traversals_to_json(traversals, output_file=str(json_out))

        result = {
            'processed': True,
            'source': str(json_path),
            'txt': str(txt_out),
            'json': str(json_out),
            'num_paths': len(filtered_paths)
        }
        return result
    except Exception as e:
        return {'error': str(e), 'source': str(json_path_str)}
    finally:
        # Best-effort lock cleanup
        try:
            # Reconstruct paths in finally since variables might be out of scope on early errors
            jp = Path(json_path_str)
            oroot = Path(output_root_str)
            droot = Path(dest_root_str)
            try:
                relp = jp.relative_to(oroot)
            except Exception:
                relp = Path(jp.name)
            ddir = droot / relp.parent
            bname = jp.stem
            lpath = ddir / f"{bname}.lock"
            if lpath.exists():
                os.remove(lpath)
        except Exception:
            pass


def main():
    """Batch-run approval-only simulations over all dialog JSON files in 'output/'.

    For each JSON file found, runs approval-only path discovery and writes:
      - A human-readable TXT of the approval-only paths
      - A structured JSON of traversal data for approval-only paths

    Outputs are written under a mirrored directory tree inside 'approval-paths/'.
    """
    parser = argparse.ArgumentParser(description="Batch approval-path extraction")
    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count(), help='Number of worker processes')
    parser.add_argument('--output-root', type=str, default='output', help='Root directory of input JSONs')
    parser.add_argument('--dest-root', type=str, default="/nfs_edlab/wschay/bg3-sim/approval-paths/", help='Destination root for outputs')
    args = parser.parse_args()

    # Clamp workers to at least 1
    workers = max(1, int(args.workers) if args.workers is not None else (os.cpu_count() or 1))

    output_root = Path(args.output_root)
    dest_root = args.dest_root

    if not output_root.exists():
        print(f"Input directory not found: {output_root.resolve()}")
        sys.exit(1)

    Path(dest_root).mkdir(parents=True, exist_ok=True)

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

    print(f"Using {workers} worker process(es)...")
    args_list = [(str(p), str(output_root), str(dest_root)) for p in json_files]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_single_file, *args) for args in args_list]
        for fut in as_completed(futures):
            res = fut.result()
            if not isinstance(res, dict):
                continue
            if 'error' in res:
                print(f"Error processing {res.get('source','?')}: {res['error']}")
                manifest['errors'].append({'source': res.get('source',''), 'error': res['error']})
                continue
            if res.get('processed'):
                files_with_approvals += 1
                total_approval_paths += int(res.get('num_paths', 0))
                per_file_counts.append((res.get('source', ''), int(res.get('num_paths', 0))))
                manifest['processed'].append({
                    'source': res.get('source', ''),
                    'txt': res.get('txt', ''),
                    'json': res.get('json', ''),
                    'num_paths': int(res.get('num_paths', 0))
                })
                print(f"Processed: {res.get('source','')} -> {res.get('num_paths',0)} paths")
                continue
            if res.get('no_approvals'):
                files_without_approvals += 1
                print(f"No approval paths: {res.get('source','')}")
                continue
            if res.get('skipped_existing'):
                print(f"Skipped (exists): {res.get('source','')}")
                continue
            if res.get('skipped_rule'):
                print(f"Skipped (rule): {res.get('source','')}")
                continue
            if res.get('skipped_in_progress'):
                print(f"Skipped (in progress): {res.get('source','')}")
                continue

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


