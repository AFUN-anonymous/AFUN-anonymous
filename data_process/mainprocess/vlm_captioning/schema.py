"""Request / result containers for the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CaptionRequest:
    """One unit of work.

    id          : unique sample id ("ep_id/interval_NNN").
    images      : ordered local image paths; the prompt template fixes the order.
    text_fields : named text inputs for the template, e.g. {"instruction": ...}.
    metadata    : pass-through metadata.
    """

    id: str
    images: List[str] = field(default_factory=list)
    text_fields: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def image_paths(self) -> List[Path]:
        return [Path(p) for p in self.images]


@dataclass
class CaptionResult:
    """Output for one request.

    status     : "done" | "error".
    raw_output : raw model text.
    parsed     : task-specific parsed fields, e.g. {"rationale", "sam3_prompt"}.
    error      : error message when status == "error".
    """

    id: str
    status: str = "done"
    raw_output: str = ""
    parsed: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
