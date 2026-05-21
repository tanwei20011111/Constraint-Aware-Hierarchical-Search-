from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExtractionRuleProfile:
    name: str
    metadata_prefixes: tuple[str, ...]
    definition_keywords: tuple[str, ...]
    window_max_chars: int
    window_overlap_paragraphs: int
    hierarchy_hints: str
    code_format_hint: str


def default_rules_dir(workspace_root: Path | None = None) -> Path:
    if workspace_root is not None:
        return workspace_root / "rules"
    return Path(__file__).resolve().parents[1] / "rules"


def normalize_dataset_rule_name(
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
) -> str | None:
    if dataset_name and str(dataset_name).strip():
        return Path(str(dataset_name).strip()).stem
    if dataset_path is None:
        return None
    return Path(dataset_path).stem.strip() or None


def dataset_rules_dir(
    dataset_name: str | None = None,
    *,
    workspace_root: Path | None = None,
    dataset_path: str | Path | None = None,
) -> Path | None:
    resolved_name = normalize_dataset_rule_name(dataset_name, dataset_path)
    if not resolved_name:
        return None
    return default_rules_dir(workspace_root) / resolved_name


def default_rule_file(
    workspace_root: Path | None = None,
    *,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
) -> Path:
    dataset_dir = dataset_rules_dir(
        dataset_name,
        workspace_root=workspace_root,
        dataset_path=dataset_path,
    )
    if dataset_dir is not None:
        dataset_rule_file = dataset_dir / "rule_profiles.json"
        if dataset_rule_file.exists():
            return dataset_rule_file
    return default_rules_dir(workspace_root) / "rule_profiles.json"


def default_rule_map_file(
    workspace_root: Path | None = None,
    *,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
) -> Path:
    dataset_dir = dataset_rules_dir(
        dataset_name,
        workspace_root=workspace_root,
        dataset_path=dataset_path,
    )
    if dataset_dir is not None:
        dataset_rule_map = dataset_dir / "rule_map.json"
        if dataset_rule_map.exists():
            return dataset_rule_map
    return default_rules_dir(workspace_root) / "rule_map.json"


def _as_profile(name: str, payload: dict[str, Any]) -> ExtractionRuleProfile:
    return ExtractionRuleProfile(
        name=name,
        metadata_prefixes=tuple(payload.get("metadata_prefixes", [])),
        definition_keywords=tuple(payload.get("definition_keywords", [])),
        window_max_chars=int(payload.get("window_max_chars", 1200)),
        window_overlap_paragraphs=int(payload.get("window_overlap_paragraphs", 1)),
        hierarchy_hints=str(payload.get("hierarchy_hints", "")),
        code_format_hint=str(payload.get("code_format_hint", "")),
    )


@lru_cache(maxsize=8)
def load_rule_profiles(rule_file: str | None = None) -> dict[str, ExtractionRuleProfile]:
    path = Path(rule_file) if rule_file else default_rule_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    return {name: _as_profile(name, body) for name, body in profiles.items()}


def load_rule_map(rule_map_path: Path | None) -> dict[str, Any]:
    if rule_map_path is None or not rule_map_path.exists():
        return {"default_rule": "default", "files": {}}
    payload = json.loads(rule_map_path.read_text(encoding="utf-8"))
    return {
        "default_rule": payload.get("default_rule", "default"),
        "files": payload.get("files", {}),
    }


def resolve_rule_name_for_file(
    filename: str,
    *,
    override_rule: str | None,
    rule_map: dict[str, Any],
) -> str:
    if override_rule:
        return override_rule
    for pattern, rule_name in rule_map.get("files", {}).items():
        if Path(filename).match(pattern):
            return str(rule_name)
    return str(rule_map.get("default_rule", "default"))
