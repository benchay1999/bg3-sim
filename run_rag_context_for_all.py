import argparse
import json
import os
import sys
from pathlib import Path

from dialog_simulator import DialogSimulator


def mirror_dest_for_json(json_path: Path, dest_root: Path, output_root: Path) -> Path:
    try:
        rel = json_path.relative_to(output_root)
    except Exception:
        # Fallback to filename only
        rel = json_path.name
    dest_path = dest_root / (rel if isinstance(rel, Path) else Path(rel))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    return dest_path


def collect_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob('*.json'))


def main():
    parser = argparse.ArgumentParser(description="Generate RAG contexts (LLM) only for sessions present in approval-paths, using a single loaded FAISS DB.")
    parser.add_argument("--output-root", default="output", help="Root directory to scan for raw JSON files (default: output)")
    parser.add_argument("--dest-root", default="qa-contexts-rag", help="Destination root for generated contexts (default: qa-contexts-rag)")
    parser.add_argument("--approval-root", default="approval-paths", help="Root directory containing *_approval_traversals.json files (default: approval-paths)")
    parser.add_argument("--cluster-index", default="goals_to_json_paths.json", help="Cluster index JSON path")
    parser.add_argument("--vector-db-dir", default="vector_db_synopses", help="Saved FAISS index directory")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="LiteLLM model id for LLM")
    parser.add_argument("--max-depth", type=int, default=50, help="Traversal depth bound for canonical turns")
    parser.add_argument("--test-mode", action="store_true", help="Ignore flag requirements during traversal")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieved contexts from FAISS")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=8000, help="LLM max tokens")
    parser.add_argument("--no-restrict-clusters", action="store_true", help="Do not restrict retrieval to current file's clusters")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if destination file already exists")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    dest_root = Path(args.dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Build the allowed set from approval-paths
    approval_root = Path(args.approval_root)
    approval_files = sorted(approval_root.rglob('*_approval_traversals.json'))
    allow_paths = []
    for appr in approval_files:
        try:
            rel = appr.relative_to(approval_root)
        except Exception:
            rel = appr.name
        rel_parent = rel.parent if isinstance(rel, Path) else Path('.')
        stem = appr.stem
        # Remove the suffix "_approval_traversals" from stem
        base_name = stem.replace('_approval_traversals', '')
        candidate = output_root / rel_parent / f"{base_name}.json"
        if candidate.is_file():
            allow_paths.append(candidate)

    if not allow_paths:
        print(f"No eligible sessions found based on approval files under {approval_root}")
        sys.exit(1)

    # Initialize simulator once to load FAISS DB once
    # Seed with an existing JSON file
    seed_json = None
    for p in allow_paths:
        if p.is_file():
            seed_json = p
            break
    if seed_json is None:
        print("Could not locate any JSON file to initialize simulator.")
        sys.exit(1)

    seed_sim = DialogSimulator(str(seed_json))
    if not seed_sim.load_faiss_from_disk(args.vector_db_dir):
        print(f"Failed to load FAISS index from {args.vector_db_dir}")
        sys.exit(2)
    # Cache loaded FAISS structures to reuse across instances
    shared_index = seed_sim._faiss_index
    shared_dim = seed_sim._faiss_dim
    shared_id_to_meta = seed_sim._faiss_id_to_meta
    shared_path_to_id = seed_sim._faiss_path_to_id
    shared_path_to_ids = seed_sim._faiss_path_to_ids

    processed = 0
    skipped = 0
    for json_path in allow_paths:
        try:
            out_path = mirror_dest_for_json(json_path, dest_root, output_root)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            # Rebind simulator to current file
            sim = DialogSimulator(str(json_path))
            # Reuse already-loaded FAISS from seed_sim
            sim._faiss_index = shared_index
            sim._faiss_dim = shared_dim
            sim._faiss_id_to_meta = shared_id_to_meta
            sim._faiss_path_to_id = shared_path_to_id
            sim._faiss_path_to_ids = shared_path_to_ids

            canonical_turns, _agg_flags, top3_samples = sim._collect_current_file_canonical(
                relevant_flag_prefixes=None,
                max_depth=args.max_depth,
                test_mode=args.test_mode,
            )
            dialogue_samples = top3_samples if top3_samples else [canonical_turns]
            synopsis = sim.metadata.get('synopsis', '')

            ctx = sim.generate_context_with_rag(
                synopsis=synopsis,
                dialogue_samples=dialogue_samples,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                cluster_index_file=args.cluster_index,
                contexts_root='qa-contexts',
                model_name=None,
                restrict_to_current_clusters=(not args.no_restrict_clusters),
                vector_db_dir=args.vector_db_dir,
            )
            if not ctx:
                print(f"[SKIP] No context generated for: {json_path}")
                continue

            payload = {
                'source': str(json_path),
                'cluster_index': args.cluster_index,
                'vector_db_dir': args.vector_db_dir,
                'model': args.model,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'top_k': args.top_k,
                'restrict_to_current_clusters': not args.no_restrict_clusters,
                'llm_input': getattr(sim, 'last_llm_input', None) or '',
                'retrieved_sessions': getattr(sim, 'last_retrieved_sessions', []) or [],
                'retrieved_synopses': getattr(sim, 'last_retrieved_synopses', []) or [],
                'context': ctx,
            }
            with open(out_path, 'w', encoding='utf-8') as f_out:
                json.dump(payload, f_out, indent=2, ensure_ascii=False)
            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} files...")
        except Exception as e:
            print(f"[ERROR] {json_path}: {e}")

    print(f"Done. Generated {processed} contexts, skipped {skipped} existing -> {dest_root}")


if __name__ == "__main__":
    main()


