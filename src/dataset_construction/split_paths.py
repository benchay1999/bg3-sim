import argparse
import re
from pathlib import Path
from typing import List, Optional


PATH_HEADER_REGEX = re.compile(r"^Path\s+(\d+):\s*$")


def split_paths_file(input_file: Path, output_dir: Optional[Path] = None) -> int:
    """Split a large approval-paths TXT into per-path TXT files.

    A path block is defined as a section that starts with a line matching
    "Path N:" at column 0 and continues until the next "Path M:" or EOF.

    The output preserves the original text exactly for each block, including
    the header line "Path N:" and all subsequent lines until (but not including)
    the next path header.

    Args:
        input_file: Path to the source TXT with many paths.
        output_dir: Optional directory to write per-path files to. If not
            provided, a directory named "<input_stem>_paths" will be created
            alongside the input file.

    Returns:
        The number of path files written.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    destination_dir = (
        output_dir
        if output_dir is not None
        else input_file.parent / f"{input_file.stem}_paths"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Stream through the file line-by-line to avoid loading entire file into memory
    current_block_lines: List[str] = []
    current_path_number: Optional[int] = None
    written_count = 0

    def flush_block() -> None:
        nonlocal current_block_lines, current_path_number, written_count
        if current_path_number is None or not current_block_lines:
            return
        filename = f"{input_file.stem}_path_{current_path_number:04d}.txt"
        out_path = destination_dir / filename
        # Write exactly what we read to preserve spacing/indentation
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(current_block_lines)
        written_count += 1
        # Reset for next block
        current_block_lines = []
        current_path_number = None

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            header_match = PATH_HEADER_REGEX.match(line)
            if header_match is not None:
                # Starting a new block. First, flush the previous one (if any).
                flush_block()
                current_path_number = int(header_match.group(1))
                current_block_lines.append(line)
            else:
                # If we are inside a block, keep collecting lines
                if current_block_lines:
                    current_block_lines.append(line)
                else:
                    # We are in the preface/header area before the first Path header.
                    # Skip these lines, as they are not part of any Path block.
                    continue

    # Flush the final block when EOF is reached
    flush_block()

    return written_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a combined approval-paths TXT file into separate per-path files."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the combined *_approval_paths.txt file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to '<input_stem>_paths' next to the input file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_file
    output_dir: Optional[Path] = args.out_dir

    written = split_paths_file(input_path, output_dir)
    dest_dir = (
        output_dir if output_dir is not None else input_path.parent / f"{input_path.stem}_paths"
    )
    print(
        f"Wrote {written} path file(s) to {dest_dir.resolve()} from {input_path.resolve()}"
    )


if __name__ == "__main__":
    main()



