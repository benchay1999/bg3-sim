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
        rel = json_path.name
    dest_path = dest_root / (rel if isinstance(rel, Path) else Path(rel))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Generate a single RAG-based context for one dialogue JSON using a prebuilt FAISS DB.")
    parser.add_argument("json_path", help="Path to the dialogue JSON (e.g., output/Act1/Goblin/GOB_NerdyGoblinSage.json)")
    parser.add_argument("--cluster-index", default="goals_to_json_paths.json", help="Cluster index JSON (default: goals_to_json_paths.json)")
    parser.add_argument("--vector-db-dir", default="vector_db_synopses", help="Directory containing saved FAISS index (default: vector_db_synopses)")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="LiteLLM model id for LLM (default: openai/gpt-5-mini)")
    parser.add_argument("--max-depth", type=int, default=50, help="Traversal depth bound when collecting canonical turns (default: 50)")
    parser.add_argument("--test-mode", action="store_true", help="Ignore flag requirements during traversal (default off)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieved contexts from FAISS (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=8000, help="LLM max tokens (default: 8000)")
    parser.add_argument("--no-restrict-clusters", action="store_true", help="Do not restrict retrieval to current file's clusters")
    parser.add_argument("--save", action="store_true", help="Save output to qa-contexts mirror path instead of printing")
    parser.add_argument("--dest-root", default="qa-contexts-rag", help="Destination root when --save is used (default: qa-contexts-rag)")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.is_file():
        print(f"Input file not found: {json_path}")
        sys.exit(1)

    sim = DialogSimulator(str(json_path))

    # Build canonical turns and samples
    canonical_turns, _agg_flags, top3_samples = sim._collect_current_file_canonical(
        relevant_flag_prefixes=None,
        max_depth=args.max_depth,
        test_mode=args.test_mode,
    )
    dialogue_samples = top3_samples if top3_samples else [canonical_turns]
    synopsis = sim.metadata.get('synopsis', '')

    # RAG using saved FAISS DB
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
        print("No context generated (check FAISS DB existence and LLM access).")
        sys.exit(2)

    if args.save:
        out_path = mirror_dest_for_json(json_path, Path(args.dest_root), Path('output'))
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
        print(f"Saved: {out_path}")
    else:
        print(ctx)


if __name__ == "__main__":
    main()


