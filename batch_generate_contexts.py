import sys
import json
import argparse
from pathlib import Path

from dialog_simulator import DialogSimulator


def main():
    """Batch-generate single, cluster-aware LLM contexts for all dialog JSON files in 'output/'.

    For each JSON file found, this script:
      1) Loads the dialogue into DialogSimulator
      2) Computes a canonical set of turns and aggregated relevant flags
      3) Uses related files from the cluster index to extend the synopsis
      4) Calls an LLM (via LiteLLM) ONCE to produce a single context per file

    Outputs are written under a mirrored directory tree inside 'qa-contexts/'.

    Requires:
      - Optional: LiteLLM installed and provider API keys in environment
      - Optional: goals_to_json_paths.json (cluster index) at project root
    """

    parser = argparse.ArgumentParser(description="Batch-generate LLM contexts for BG3 dialogues.")
    parser.add_argument("--output-root", default="output", help="Root directory to scan for raw JSON files.")
    parser.add_argument("--dest-root", default="qa-contexts", help="Destination root for generated context files.")
    parser.add_argument("--cluster-index", default="goals_to_json_paths.json", help="Cluster index JSON file path.")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="LiteLLM model id (e.g., 'openai/gpt-5-mini' or 'gemini/gemini-2.5-flash').")
    parser.add_argument("--max-depth", type=int, default=50, help="Traversal depth bound for canonical turn discovery.")
    parser.add_argument("--test-mode", action="store_true", help="Ignore flag requirements during traversal (default off).")
    parser.add_argument("--prefix", action="append", default=[], help="Relevant flag prefix filter; can be repeated. Example: --prefix ORI_INCLUSION_ --prefix QUEST_")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="LLM max tokens for completion.")
    parser.add_argument("--skip", action="append", default=["SHA_Merregon_000.json", "LOW_StormshoreTabernacle_TyrShrine.json", "CAMP_Courier_Dog.json"], help="File basenames to skip. Can repeat.")

    args = parser.parse_args()

    output_root = Path(args.output_root)
    dest_root = Path(args.dest_root)

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

    for json_path in json_files:
        try:
            rel_path = json_path.relative_to(output_root)
        except ValueError:
            rel_path = json_path.name

        base_name = json_path.name
        if any(skip_name in base_name for skip_name in args.skip):
            continue

        print(f"\n=== Generating context: {json_path} ===")

        try:
            simulator = DialogSimulator(str(json_path))

            # Build context once per file, using cluster-aware RAG by default
            ctx = simulator.generate_cluster_context_for_current_file(
                cluster_index_file=args.cluster_index,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                relevant_flag_prefixes=(args.prefix if args.prefix else None),
                max_depth=args.max_depth,
                test_mode=args.test_mode,
                verbose=False
            )

            if not ctx:
                manifest['errors'].append({'source': str(json_path), 'error': 'Empty context or LLM unavailable'})
                print("  -> No context generated (possibly missing LiteLLM or API key).")
                continue

            dest_dir = dest_root / (rel_path.parent if isinstance(rel_path, Path) else Path('.'))
            dest_dir.mkdir(parents=True, exist_ok=True)

            stem = json_path.stem
            txt_out = dest_dir / f"{stem}_context.txt"
            meta_out = dest_dir / f"{stem}_context_meta.json"

            with open(txt_out, 'w', encoding='utf-8') as f_txt:
                f_txt.write(ctx)

            with open(meta_out, 'w', encoding='utf-8') as f_meta:
                json.dump({
                    'source': str(json_path),
                    'model': args.model,
                    'temperature': args.temperature,
                    'max_tokens': args.max_tokens,
                    'relevant_flag_prefixes': args.prefix or [],
                    'cluster_index': args.cluster_index,
                    'llm_input': getattr(simulator, 'last_llm_input', None) or '',
                }, f_meta, indent=2, ensure_ascii=False)

            manifest['processed'].append({
                'source': str(json_path),
                'context_txt': str(txt_out),
                'meta_json': str(meta_out)
            })

        except Exception as e:
            manifest['errors'].append({'source': str(json_path), 'error': str(e)})
            print(f"  -> Error: {e}")

    # Write manifest
    manifest_path = dest_root / '_contexts_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote context manifest: {manifest_path}")


if __name__ == '__main__':
    main()


