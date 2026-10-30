from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit arriving on the stream. `ts` is *event time* (epoch seconds)."""
    id: str
    text: str
    ts: float
    title: str = ""
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text,
                "ts": self.ts, "source": self.source, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Document":
        return Document(id=str(d["id"]), text=d.get("text", ""), ts=float(d["ts"]),
                        title=d.get("title", ""), source=d.get("source", "unknown"),
                        meta=d.get("meta", {}) or {})


@dataclass
class Hit:
    doc_id: str
    score: float
    title: str
    snippet: str
    ts: float
    source: str
    explain: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.doc_id, "score": round(self.score, 5), "title": self.title,
                "snippet": self.snippet, "ts": self.ts, "source": self.source,
                "explain": self.explain}
