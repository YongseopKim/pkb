"""Data models for vocabulary (domains & topics)."""

from typing import Literal

from pydantic import BaseModel


class Domain(BaseModel):
    """An L1 domain category."""

    id: str
    label_ko: str
    label_en: str


class DomainsVocab(BaseModel):
    """Collection of L1 domains."""

    domains: list[Domain]

    def get_ids(self) -> set[str]:
        """Return the set of all domain IDs."""
        return {d.id for d in self.domains}


class Topic(BaseModel):
    """An L2 topic entry in controlled vocabulary."""

    canonical: str
    aliases: list[str] = []
    status: Literal["approved", "pending", "merged"] = "approved"
    merged_into: str | None = None


class TopicsVocab(BaseModel):
    """Collection of L2 topics."""

    topics: list[Topic]

    def get_approved_canonicals(self) -> set[str]:
        """Return only approved topic canonical names."""
        return {t.canonical for t in self.topics if t.status == "approved"}
