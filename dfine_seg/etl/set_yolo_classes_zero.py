from __future__ import annotations

import argparse
import re
from pathlib import Path

LINE_PATTERN = re.compile(r"^(\s*)\S+(\s+.*)?$")


def rewrite_line(line: str) -> str:
    if not line.strip():
        return line

    line_body = line
    line_ending = ""
    if line.endswith("\r\n"):
        line_body = line[:-2]
        line_ending = "\r\n"
    elif line.endswith("\n"):
        line_body = line[:-1]
        line_ending = "\n"

    match = LINE_PATTERN.match(line_body)
    if match is None:
        raise ValueError(f"Unsupported label line: {line_body!r}")

    return f"{match.group(1)}0{match.group(2) or ''}{line_ending}"


def rewrite_file(label_path: Path) -> tuple[bool, int]:
    original_lines = label_path.read_text(encoding="utf-8").splitlines(keepends=True)
    rewritten_lines = []
    changed = False
    annotation_lines = 0

    for line in original_lines:
        rewritten_line = rewrite_line(line)
        if line.strip():
            annotation_lines += 1
        if rewritten_line != line:
            changed = True
        rewritten_lines.append(rewritten_line)

    if changed:
        label_path.write_text("".join(rewritten_lines), encoding="utf-8")

    return changed, annotation_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite every YOLO label file so all class ids become 0."
    )
    parser.add_argument("labels_dir", type=Path, help="Root directory containing .txt label files")
    args = parser.parse_args()

    labels_dir = args.labels_dir.expanduser().resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    total_files = 0
    changed_files = 0
    total_annotations = 0

    for label_path in labels_dir.rglob("*.txt"):
        total_files += 1
        file_changed, annotation_lines = rewrite_file(label_path)
        total_annotations += annotation_lines
        changed_files += int(file_changed)

        if total_files % 10000 == 0:
            print(
                f"Processed {total_files} files | changed {changed_files} | annotations {total_annotations}"
            )

    print(f"Processed {total_files} files")
    print(f"Changed {changed_files} files")
    print(f"Seen {total_annotations} annotation lines")


if __name__ == "__main__":
    main()
