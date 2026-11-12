"""Multi-level label taxonomy: visual class / instance identity / client code / context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() in {"", "NULL", "None", "nan"}


def to_float(value: Any) -> float | None:
    """Parse numeric attributes that the client sometimes stores as text."""
    if _blank(value):
        return None
    try:
        out = float(str(value).replace(",", "."))
    except ValueError:
        return None
    return out if math.isfinite(out) else None


@dataclass(frozen=True)
class LabelDecision:
    visual_class: str | None
    visual_id: int | None
    client_code: str | None
    client_name: str | None
    flags: tuple[str, ...]


class Taxonomy:
    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.version = raw["version"]
        self.background_id = int(raw["background_id"])
        self.ignore_id = int(raw["ignore_id"])
        self.visual = {c["name"]: int(c["id"]) for c in raw["visual_classes"]}
        ids = list(self.visual.values())
        if len(set(ids)) != len(ids) or {self.background_id, self.ignore_id} & set(ids):
            raise ValueError("Visual class ids must be unique and exclude background/ignore")
        self.longitudinal = raw["longitudinal"]
        self.other = raw["other_marks"]
        self.policy = raw["label_policy"]

    @classmethod
    def load(cls, path: str | Path) -> Taxonomy:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def id_to_name(self) -> dict[int, str]:
        return {v: k for k, v in self.visual.items()}

    def classify_line(self, attrs: dict) -> LabelDecision:
        flags: list[str] = []
        text = attrs.get(self.longitudinal["designation_field"])
        entry = None if _blank(text) else self.longitudinal["designations"].get(str(text).strip())
        if entry is None:
            flags.append("unknown_designation" if not _blank(text) else "missing_designation")
            return LabelDecision(
                None, None, None, None if _blank(text) else str(text), tuple(flags)
            )
        code = entry.get("code")
        if entry.get("ambiguous"):
            flags.append("client_code_ambiguous")
        visual = None
        if entry.get("visual_rule") == "dash_if_traco":
            traco = to_float(attrs.get("TRAÇO"))
            visual = "longitudinal_dash" if traco is not None else "longitudinal_solid"
        else:
            flags.append("no_visual_rule")
        return LabelDecision(
            visual,
            self.visual.get(visual) if visual else None,
            None if code is None else str(code),
            entry.get("name"),
            tuple(flags),
        )

    def classify_polygon(self, attrs: dict) -> LabelDecision:
        flags: list[str] = []
        for field in self.other.get("character_fields", []):
            if not _blank(attrs.get(field)):
                flags.append("character_label_unconfirmed")
                return LabelDecision(
                    "character",
                    self.visual["character"],
                    str(attrs.get(field)),
                    field,
                    tuple(flags),
                )
        code = attrs.get(self.other["code_field"])
        if _blank(code):
            return LabelDecision(None, None, None, None, ("missing_class_code",))
        entry = self.other["codes"].get(str(code).strip())
        if entry is None:
            return LabelDecision(None, None, str(code), None, ("unknown_class_code",))
        if entry.get("to_confirm"):
            flags.append("catalog_mapping_to_confirm")
        visual = entry["visual"]
        return LabelDecision(visual, self.visual[visual], str(code).strip(), None, tuple(flags))

    def mapping_table(self) -> dict:
        """Machine-readable mapping between all four levels (written to metadata)."""
        return {
            "version": self.version,
            "provisional": bool(self.raw.get("provisional", True)),
            "background_id": self.background_id,
            "ignore_id": self.ignore_id,
            "visual_classes": self.raw["visual_classes"],
            "instance_identity": (
                "global_id = '<source file name>|<layer>|fid=<fid>'; integer instance ids are "
                "assigned by sorted global_id; per-tile local ids 1..N are contiguous"
            ),
            "longitudinal_designations": self.longitudinal["designations"],
            "unmapped_longitudinal_codes": self.longitudinal.get("unmapped_client_codes", []),
            "other_mark_codes": self.other["codes"],
            "context_attributes": {
                "longitudinal": self.longitudinal.get("context_attributes", []),
                "other_marks": self.other.get("context_attributes", []),
            },
            "label_policy": self.policy,
        }
