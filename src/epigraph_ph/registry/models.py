from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class LiteratureRefDetail:
    source_id: str
    title: str | None = None
    year: int | None = None
    source_tier: str | None = None
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    openalex_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_verifiable_locator(item: dict[str, Any]) -> bool:
    return any(str(item.get(key) or "").strip() for key in ("url", "doi", "pmid", "openalex_id"))
