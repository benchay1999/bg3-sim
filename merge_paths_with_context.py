import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PATH_HEADER_REGEX = re.compile(r"^Path\s+(\d+):\s*$")


def _relative_under_approval_paths(p: Path) -> Optional[Path]:
    """Return the subpath under 'approval-paths' if present in the path.

    Works for both absolute and relative paths. Returns None if 'approval-paths'
    segment is not present in the path parts.
    """
    parts = p.parts
    try:
        idx = parts.index("approval-paths")
    except ValueError:
        return None
    # Join the components after 'approval-paths/'
    return Path(*parts[idx + 1 :])


def parse_paths_from_txt(approval_paths_file: Path) -> List[Dict[str, object]]:
    """Parse an approval paths TXT into a list of path blocks.

    Each block begins with a header line of the form "Path N:" at column 0 and
    continues until the next header or end of file. The returned text for each
    block preserves the exact original content (including the header line).
    """
    paths: List[Dict[str, object]] = []
    current_lines: List[str] = []
    current_number: Optional[int] = None

    def flush() -> None:
        nonlocal current_lines, current_number
        if current_number is None or not current_lines:
            return
        text_list = "".join(current_lines).split("\n")
        processed_text_list = []
        for item in text_list[1:]:
            add_flag = not ": True" in item and not ": False" in item and item != ""
            if "[approval]" in item:
                add_flag = True
            if add_flag:
                processed_text_list.append(item)
        paths.append({
            "number": current_number,
            "text": processed_text_list,
        })
        current_lines = []
        current_number = None

    with open(approval_paths_file, "r", encoding="utf-8") as f:
        for line in f:
            m = PATH_HEADER_REGEX.match(line)
            if m is not None:
                flush()
                current_number = int(m.group(1))
                current_lines.append(line)
            else:
                if current_lines:
                    current_lines.append(line)
                else:
                    # Skip preface lines before the first path header
                    continue

    flush()
    return paths


def infer_context_path(approval_paths_file: Path, contexts_root: Path) -> Path:
    """Infer the context JSON path in qa-contexts-rag from an approval paths file path.

    Mirrors directory under approval-paths/, replaces suffix _approval_paths.txt with .json.
    Example: approval-paths/Act1/Chapel/NAME_approval_paths.txt -> qa-contexts-rag/Act1/Chapel/NAME.json
    """
    rel = _relative_under_approval_paths(approval_paths_file)
    if rel is not None:
        base = rel.name
        dir_part = rel.parent
    else:
        base = approval_paths_file.name
        dir_part = Path("")

    if not base.endswith("_approval_paths.txt"):
        raise ValueError(
            f"Unexpected approval paths filename (must end with _approval_paths.txt): {approval_paths_file}"
        )
    context_name = base.replace("_approval_paths.txt", ".json")
    return contexts_root / dir_part / context_name


def load_context_from_json(context_json_path: Path) -> str:
    """Load the 'context' field from a context JSON saved in qa-contexts-rag."""
    with open(context_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ctx = data.get("context", "")
    return ctx or ""


def insert_context_after_metadata(approval_paths_file: Path, context_text: str) -> bool:
    """Insert a Context: block after metadata (before first Path ...) in the TXT.

    Returns True if file was modified, False if skipped (e.g., context already present).
    """
    with open(approval_paths_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find first Path header line index
    first_path_idx = None
    for i, line in enumerate(lines):
        if PATH_HEADER_REGEX.match(line):
            first_path_idx = i
            break

    if first_path_idx is None:
        # No paths; treat entire file as metadata
        first_path_idx = len(lines)

    # Detect existing Context: block in metadata region
    if any(l.strip().startswith("Context:") for l in lines[:first_path_idx]):
        return False

    block = ["Context:\n", context_text.rstrip() + "\n", "\n"]
    new_lines = lines[:first_path_idx] + block + lines[first_path_idx:]
    with open(approval_paths_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    return True


def merge_one(
    approval_paths_file: Path,
    context_file: Optional[Path],
    output_file: Optional[Path],
    results_root: Optional[Path] = None,
) -> Tuple[Path, int]:
    """Merge a pair of files into a single JSON with context and paths.

    If context_file is None, will infer it from the approval_paths_file path assuming
    qa-contexts/ mirrors approval-paths/.
    If output_file is None, writes next to the approval_paths_file with suffix
    "_with_context.json".
    Returns the output path and number of paths merged.
    """
    if not approval_paths_file.exists():
        raise FileNotFoundError(f"Approval paths file not found: {approval_paths_file}")

    if context_file is None:
        context_file = infer_context_path(approval_paths_file, Path("qa-contexts"))

    if not context_file.exists():
        raise FileNotFoundError(
            f"Context file not found for {approval_paths_file}: {context_file}"
        )

    if output_file is None:
        # Default to mirrored path under results_root (if provided), else next to input
        if results_root is not None:
            rel = _relative_under_approval_paths(approval_paths_file)
            if rel is not None:
                out_dir = results_root / rel.parent
            else:
                out_dir = results_root
            output_file = out_dir / f"{approval_paths_file.stem}_with_context.json"
        else:
            output_file = approval_paths_file.with_name(
                f"{approval_paths_file.stem}_with_context.json"
            )

    paths = parse_paths_from_txt(approval_paths_file)
    context_text = load_context_from_json(context_file)

    output_payload = {
        "source_approval_paths": str(approval_paths_file),
        "source_context": str(context_file),
        "num_paths": len(paths),
        "context": context_text,
        "paths": paths,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    return output_file, len(paths)


def batch_merge(
    approval_root: Path, contexts_root: Path, results_root: Path
) -> List[Tuple[Path, int]]:
    """Recursively find all approval path TXT files and merge with contexts.

    Returns a list of (output_file, num_paths) for successful merges. Skips files
    that do not have matching contexts.
    """
    results: List[Tuple[Path, int]] = []
    for txt in sorted(approval_root.rglob("*_approval_paths.txt")):
        try:
            context_file = infer_context_path(txt, contexts_root)
            if not context_file.exists():
                print(f"[skip] No context found for {txt}")
                continue
            # Insert context block into the TXT after metadata (idempotent)
            try:
                ctx_str = load_context_from_json(context_file)
                if ctx_str:
                    modified = insert_context_after_metadata(txt, ctx_str)
                    if modified:
                        print(f"[updated] Inserted context into {txt}")
            except Exception as _e:
                pass
            # Compute output path under results_root, mirroring path under 'approval-paths/'
            rel_global = _relative_under_approval_paths(txt)
            out_dir = (
                results_root / rel_global.parent if rel_global is not None else results_root
            )
            out_path = out_dir / f"{txt.stem}_with_context.json"
            res = merge_one(txt, context_file, out_path, results_root)
            results.append(res)
            print(f"[ok] {txt} -> {res[0]} ({res[1]} paths)")

            # Write per-path JSON files under results/<rel_dir>/<name>_path_xxxx.json
            try:
                merged = json.loads(out_path.read_text(encoding="utf-8"))
                rel_dir = _relative_under_approval_paths(txt)
                target_dir = results_root / rel_dir.parent if rel_dir is not None else results_root
                base = txt.stem.replace("_approval_paths", "")
                for path_obj in merged.get("paths", []):
                    num = path_obj.get("number")
                    if num is None:
                        continue
                    fname = f"{base}_path_{num:04d}.json"
                    payload = {
                        "source_approval_paths": merged.get("source_approval_paths"),
                        "context": merged.get("context", ""),
                        "path_number": num,
                        "text": path_obj.get("text", []),
                    }
                    (target_dir / fname).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as _e:
                pass
        except Exception as e:
            print(f"[error] {txt}: {e}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge approval-paths TXT files with matching qa-contexts TXT into JSON."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--approval",
        type=Path,
        help="Path to a single *_approval_paths.txt to merge",
    )
    group.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all files under approval-paths/ recursively",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="Path to the matching *_context.txt (optional; inferred if omitted)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Explicit output JSON path when using --approval. Defaults to "
            "<approval_stem>_with_context.json next to the approval file."
        ),
    )
    parser.add_argument(
        "--approval-root",
        type=Path,
        default=Path("approval-paths"),
        help="Root directory for approval paths when using --batch",
    )
    parser.add_argument(
        "--contexts-root",
        type=Path,
        default=Path("qa-contexts-rag"),
        help="Root directory for contexts JSON (qa-contexts-rag) when using --batch or inferring",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory under which merged JSON files will be written",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.batch:
        results = batch_merge(args.approval_root, args.contexts_root, args.results_root)
        total = sum(n for _, n in results)
        print(f"Merged {len(results)} files, total {total} paths.")
        return

    # Single-file mode
    out_path, num_paths = merge_one(
        args.approval, args.context, args.out, args.results_root
    )
    print(f"Wrote {num_paths} paths to {out_path.resolve()}")


if __name__ == "__main__":
    main()


