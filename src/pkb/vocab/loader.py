"""Vocab YAML loaders."""

from pathlib import Path

import yaml

from pkb.models.vocab import DomainsVocab, TopicsVocab

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_domains(path: Path | None = None) -> DomainsVocab:
    """Load domains vocabulary from YAML.

    If no path is given, loads from bundled data.
    """
    if path is None:
        path = _DATA_DIR / "domains.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DomainsVocab(**raw)


def load_topics(path: Path | None = None) -> TopicsVocab:
    """Load topics vocabulary from YAML.

    If no path is given, loads from bundled data.
    """
    if path is None:
        path = _DATA_DIR / "topics.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TopicsVocab(**raw)
