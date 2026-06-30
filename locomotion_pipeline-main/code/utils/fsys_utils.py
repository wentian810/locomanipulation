from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


class RelativeFileMatcher:
    """Find files under a root folder and return paths relative to that root."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {self.root_dir}")

    def find(
        self,
        filename_pattern: str,
        use_regex: bool = False,
        recursive: bool = True,
        match_target: Literal["file", "dir", "both"] = "file",
    ) -> list[str]:
        """
        Find matching files/directories by name and return relative paths.

        Args:
            filename_pattern: Exact name or regular expression pattern.
            use_regex: If True, treat filename_pattern as regex.
            recursive: If True, search all subdirectories.
            match_target: Match target type, one of "file", "dir", "both".

        Returns:
            A sorted list of relative file paths (POSIX style).
        """
        if match_target not in {"file", "dir", "both"}:
            raise ValueError("match_target must be one of: 'file', 'dir', 'both'")

        if use_regex:
            pattern = re.compile(filename_pattern)

            def is_match(file_name: str) -> bool:
                return pattern.search(file_name) is not None

        else:

            def is_match(file_name: str) -> bool:
                return file_name == filename_pattern

        entries = self.root_dir.rglob("*") if recursive else self.root_dir.glob("*")
        matched_rel_paths: list[str] = []

        for path in entries:
            if match_target == "file" and not path.is_file():
                continue
            if match_target == "dir" and not path.is_dir():
                continue
            if is_match(path.name):
                matched_rel_paths.append(path.relative_to(self.root_dir).as_posix())

        return sorted(matched_rel_paths)

    def print_matches(
        self,
        filename_pattern: str,
        use_regex: bool = False,
        recursive: bool = True,
        match_target: Literal["file", "dir", "both"] = "file",
    ) -> None:
        """Print matched relative paths line by line."""
        for rel_path in self.find(
            filename_pattern=filename_pattern,
            use_regex=use_regex,
            recursive=recursive,
            match_target=match_target,
        ):
            print(rel_path)

    def export_matches(
        self,
        output_file: str | Path,
        filename_pattern: str,
        use_regex: bool = False,
        recursive: bool = True,
        match_target: Literal["file", "dir", "both"] = "file",
        encoding: str = "utf-8",
    ) -> list[str]:
        """Export matched relative paths to a text file, one path per line."""
        matches = self.find(
            filename_pattern=filename_pattern,
            use_regex=use_regex,
            recursive=recursive,
            match_target=match_target,
        )
        output_path = Path(output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(matches), encoding=encoding)
        return matches

if __name__ == "__main__":
    matcher = RelativeFileMatcher("/media/linycs/ssd1T/projects_dataset/data_deliver/smpl2qiaojie/data/storage/input_struct")
    
    matcher.print_matches(
        filename_pattern="104010_A_Day_In_The_Life_of_a[_0-9]+",
        use_regex=True,
        match_target="both",
    )
