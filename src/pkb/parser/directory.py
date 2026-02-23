"""Directory-level parsing for JSONL and MD files."""

from pathlib import Path

from pkb.models.jsonl import Conversation
from pkb.parser.exceptions import ParseError
from pkb.parser.jsonl_parser import parse_jsonl_file
from pkb.parser.md_parser import parse_md_file

SUPPORTED_EXTENSIONS = frozenset({".jsonl", ".md"})


def find_jsonl_files(directory: Path) -> list[Path]:
    """Find all .jsonl files in a directory (non-recursive)."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(directory.glob("*.jsonl"))


def find_input_files(directory: Path) -> list[Path]:
    """Find all supported input files (.jsonl, .md) in a directory (non-recursive)."""
    from pkb.constants import SKIP_FILENAMES

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        for f in directory.glob(f"*{ext}"):
            if f.name.lower() not in SKIP_FILENAMES:
                files.append(f)
    return sorted(files)


def find_input_files_recursive(directory: Path) -> list[Path]:
    """Find all supported input files (.jsonl, .md) recursively, excluding .done/ dirs.

    Used by batch processing and watch initial scan.
    """
    from pkb.constants import DONE_DIR_NAME, SKIP_FILENAMES

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        for f in directory.rglob(f"*{ext}"):
            if DONE_DIR_NAME not in f.relative_to(directory).parts:
                if f.name.lower() not in SKIP_FILENAMES:
                    files.append(f)
    return sorted(files)


def parse_file(path: Path) -> Conversation:
    """Parse a single file by dispatching to the correct parser based on extension.

    Raises:
        ParseError: If file extension is not supported.
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        return parse_jsonl_file(path)
    elif path.suffix == ".md":
        return parse_md_file(path)
    else:
        raise ParseError(f"Unsupported file extension: {path.suffix}")


def parse_directory(directory: Path) -> list[Conversation]:
    """Parse all supported files in a directory, sorted by exported_at."""
    files = find_input_files(directory)
    conversations = [parse_file(f) for f in files]
    conversations.sort(key=lambda c: c.meta.exported_at)
    return conversations
