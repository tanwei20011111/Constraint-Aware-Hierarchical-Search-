from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regtree_agent.config import Settings
from regtree_agent.online import OnlineClients
from regtree_agent.prompts import (
    RULE_ANALYSIS_CONSTRAINTS,
    RULE_ANALYSIS_SYSTEM_PROMPT,
    RULE_ANALYSIS_TASK,
    RULE_ENGINE_CONTRACT_IMPORTANT_LIMITS,
)


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".htm",
    ".log",
}

RULE_PROFILE_FIELDS = {
    "metadata_prefixes": [],
    "definition_keywords": [],
    "window_max_chars": 1200,
    "window_overlap_paragraphs": 1,
    "hierarchy_hints": "",
    "code_format_hint": "",
}


def _progress(message: str) -> None:
    print(f"[analyze_rules] {message}", file=sys.stderr, flush=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_cli_path(value: str | None) -> Path | None:
    if not value:
        return None
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    cwd_path = raw_path.resolve()
    if cwd_path.exists():
        return cwd_path

    root_path = (ROOT / raw_path).resolve()
    if root_path.exists():
        return root_path

    return cwd_path


def _normalize_dataset_name(
    dataset_name: str | None,
    dataset_path: Path | None,
    input_path: Path | None,
    manifest_path: Path | None,
) -> str:
    if dataset_name and dataset_name.strip():
        return dataset_name.strip()
    if dataset_path is not None:
        return dataset_path.stem.strip()
    if input_path is not None:
        if input_path.name == "manifest.json":
            parent_name = input_path.parent.name
        else:
            parent_name = input_path.stem if input_path.is_file() else input_path.name
        if parent_name.endswith("_chapters"):
            return parent_name[: -len("_chapters")]
        if parent_name.endswith("_pages_4"):
            return parent_name[: -len("_pages_4")]
        return parent_name.strip()
    if manifest_path is None:
        return "dataset"
    parent_name = manifest_path.parent.name
    if parent_name.endswith("_chapters"):
        return parent_name[: -len("_chapters")]
    if parent_name.endswith("_pages_4"):
        return parent_name[: -len("_pages_4")]
    return parent_name.strip()


def _resolve_manifest(dataset_name: str | None, dataset_path: Path | None, manifest_path: Path | None) -> Path | None:
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
        return manifest_path

    candidates: list[Path] = []
    names = [name for name in [dataset_name, dataset_path.stem if dataset_path else None] if name]
    for name in names:
        candidates.append(ROOT / "data" / f"{name}_chapters" / "manifest.json")
        candidates.append(ROOT / "data" / f"{name}_pages_4" / "manifest.json")
    candidates.append(ROOT / "input" / "manifest.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_output_path(manifest_path: Path, output_file: str) -> Path:
    local_path = manifest_path.parent / Path(output_file).name
    if local_path.exists():
        return local_path
    absolute_path = Path(output_file)
    if absolute_path.exists():
        return absolute_path
    raise FileNotFoundError(f"找不到 manifest 对应 chunk 文件: {output_file}")


def _load_entries(manifest_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(manifest_path)
    if not isinstance(payload, list):
        raise TypeError(f"manifest 必须是数组，当前是 {type(payload).__name__}")
    entries = [item for item in payload if isinstance(item, dict)]
    if not entries:
        raise ValueError(f"manifest 没有可用条目: {manifest_path}")
    return entries


def _sample_entries(entries: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    if len(entries) <= sample_count:
        return entries
    indices = {0, len(entries) - 1}
    for i in range(sample_count):
        idx = round(i * (len(entries) - 1) / max(1, sample_count - 1))
        indices.add(idx)
    return [entries[idx] for idx in sorted(indices)][:sample_count]


def _truncate(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


_STRUCTURE_LINE_RE = re.compile(
    r"^\s*(\d+[\.\)]\s|第.+[章编节类部篇]\s|[A-Z]\.\s|[(（]\d+\s|\d{2}\.\d{2}\s{2})",
)


def _extract_structure_lines(text: str, max_chars: int) -> str:
    """优先提取包含编号、标题格式的行，其余行按可用空间补充。"""
    lines = text.splitlines()
    structure_lines: list[str] = []
    other_lines: list[str] = []
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if _STRUCTURE_LINE_RE.match(clean):
            structure_lines.append(clean)
        else:
            other_lines.append(clean)
    result = "\n".join(structure_lines)
    remaining = max_chars - len(result) - 3
    if remaining > 50 and other_lines:
        result += "\n" + "\n".join(other_lines[: remaining // 40])
    if len(result) > max_chars:
        result = result[: max_chars - 3] + "..."
    return result


def _build_samples(
    manifest_path: Path,
    entries: list[dict[str, Any]],
    *,
    sample_count: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for entry in _sample_entries(entries, sample_count):
        output_path = _normalize_output_path(manifest_path, str(entry.get("output_file", "")))
        text = output_path.read_text(encoding="utf-8")
        samples.append(
            {
                "index": entry.get("index"),
                "title": str(entry.get("title", "")).strip(),
                "start_page": entry.get("start_page"),
                "end_page": entry.get("end_page"),
                "chunk_file": output_path.name,
                "content_preview": _extract_structure_lines(text, max_chars),
            }
        )
    return samples


def _iter_text_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files: list[Path] = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.name == "manifest.json":
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
    if files:
        return files

    # Fallback: allow extensionless text files. Binary files will be skipped by
    # _read_text_file.
    return [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and not path.name.startswith(".") and path.name != "manifest.json"
    ]


def _read_text_file(path: Path) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
        if "\x00" in text:
            return None
        return text
    return None


def _build_content_samples(
    input_path: Path,
    *,
    sample_count: int,
    max_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    files = _iter_text_files(input_path)
    readable: list[tuple[Path, str]] = []
    for path in files:
        text = _read_text_file(path)
        if text and text.strip():
            readable.append((path, text))
    if not readable:
        raise ValueError(f"没有从输入路径读取到可分析的文本文件: {input_path}")

    sampled = _sample_entries(
        [{"index": i, "path": str(path), "text": text} for i, (path, text) in enumerate(readable, start=1)],
        sample_count,
    )
    samples: list[dict[str, Any]] = []
    for item in sampled:
        path = Path(str(item["path"]))
        samples.append(
            {
                "index": item["index"],
                "title": path.stem,
                "source_file": str(path),
                "content_preview": _truncate(str(item["text"]), max_chars),
            }
        )
    return samples, len(readable)


def _load_base_profiles(base_rule_file: Path | None) -> dict[str, dict[str, Any]]:
    if base_rule_file is None:
        return {}
    profile_files = [base_rule_file]
    profiles: dict[str, dict[str, Any]] = {}
    for profile_file in profile_files:
        if profile_file is None or not profile_file.exists():
            continue
        payload = _read_json(profile_file)
        raw_profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
        for name, profile in raw_profiles.items():
            if isinstance(profile, dict):
                profiles[str(name)] = profile
    return profiles


def _choose_base_profile(
    profiles: dict[str, dict[str, Any]],
    base_profile_name: str | None,
) -> dict[str, Any]:
    if base_profile_name:
        if base_profile_name not in profiles:
            raise KeyError(f"基础 profile 不存在: {base_profile_name}; 可用: {', '.join(profiles)}")
        return profiles[base_profile_name]
    if profiles:
        return next(iter(profiles.values()))
    return RULE_PROFILE_FIELDS


def _ensure_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    result = [str(item).strip() for item in value if str(item).strip()]
    return result or fallback


def _ensure_int(value: Any, fallback: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, parsed)



def _normalize_rule_profile(generated: dict[str, Any], base_profile: dict[str, Any]) -> dict[str, Any]:
    raw = generated.get("rule_profile", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "metadata_prefixes": _ensure_string_list(
            raw.get("metadata_prefixes"),
            list(base_profile.get("metadata_prefixes", RULE_PROFILE_FIELDS["metadata_prefixes"])),
        ),
        "definition_keywords": _ensure_string_list(
            raw.get("definition_keywords"),
            list(base_profile.get("definition_keywords", RULE_PROFILE_FIELDS["definition_keywords"])),
        ),
        "window_max_chars": _ensure_int(
            raw.get("window_max_chars"),
            int(base_profile.get("window_max_chars", RULE_PROFILE_FIELDS["window_max_chars"])),
            200,
        ),
        "window_overlap_paragraphs": _ensure_int(
            raw.get("window_overlap_paragraphs"),
            int(base_profile.get("window_overlap_paragraphs", RULE_PROFILE_FIELDS["window_overlap_paragraphs"])),
            0,
        ),
        "hierarchy_hints": str(
            raw.get("hierarchy_hints", base_profile.get("hierarchy_hints", ""))
        ).strip(),
        "code_format_hint": str(
            raw.get("code_format_hint", base_profile.get("code_format_hint", ""))
        ).strip(),
    }


def _normalize_rule_map(dataset_name: str, generated: dict[str, Any]) -> dict[str, Any]:
    raw = generated.get("rule_map", {})
    raw_files = raw.get("files", {}) if isinstance(raw, dict) else {}
    files = {
        str(pattern).strip(): str(rule_name).strip()
        for pattern, rule_name in raw_files.items()
        if str(pattern).strip() and str(rule_name).strip()
    }
    return {"default_rule": dataset_name, "files": files}


def _build_engine_contract() -> dict[str, Any]:
    return {
        "rule_profile_fields": RULE_PROFILE_FIELDS,
        "important_limits": RULE_ENGINE_CONTRACT_IMPORTANT_LIMITS,
    }


def _build_output_schema(dataset_name: str) -> dict[str, Any]:
    return {
        "analysis": {
            "document_structure": "文档整体结构说明",
            "metadata_lines": "样本中是否存在标题、来源、页码等元数据行；如果没有就说明无",
            "observed_heading_examples": ["从样本原文摘录的主条目示例；没有则空数组"],
            "observed_subheading_examples": ["从样本原文摘录的子项示例；没有则空数组"],
            "numbering_strategy": "说明如何从样本中的编号、标题、缩进或标点推导任意深度层级；没有稳定编号则说明无",
            "risks": ["可能误识别的地方"],
        },
        "rule_profile": RULE_PROFILE_FIELDS,
        "rule_map": {"default_rule": dataset_name, "files": {}},
    }


def _build_constraints() -> list[str]:
    return RULE_ANALYSIS_CONSTRAINTS


def _build_prompt(
    *,
    dataset_name: str,
    source_description: dict[str, Any],
    sample_titles: list[str],
    total_sources: int,
    base_profiles: dict[str, dict[str, Any]],
    base_profile: dict[str, Any],
    samples: list[dict[str, Any]],
) -> str:
    payload = {
        "task": RULE_ANALYSIS_TASK,
        "dataset_name": dataset_name,
        "source": source_description,
        "source_count": total_sources,
        "sample_titles": sample_titles[:50],
        "current_engine_contract": _build_engine_contract(),
        "base_profile_for_optional_reference": base_profile,
        "available_base_profile_names": list(base_profiles.keys()),
        "sample_chunks": samples,
        "output_schema": _build_output_schema(dataset_name),
        "constraints": _build_constraints(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM to analyze document content and write regtree rule files")
    parser.add_argument("--input-path", help="Text file or directory to analyze; manifest.json not required")
    parser.add_argument("--dataset-path", help="Original dataset file path for deriving rule name")
    parser.add_argument("--dataset-name", help="Dataset name used as rule name and output dir (rules/<name>/); must match build --dataset-name; defaults to dataset-path stem or manifest dir")
    parser.add_argument("--manifest-path", help="Optional manifest.json path; reads chunks via manifest when provided")
    parser.add_argument("--base-rule-file", help="Optional reference rule_profiles.json")
    parser.add_argument("--base-profile", help="Optional reference profile name")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of chunks to sample")
    parser.add_argument("--max-chunk-chars", type=int, default=2200, help="Max chars per sample sent to the model")
    parser.add_argument("--output-root", default=str(ROOT / "rules"), help="Rule output root directory")
    parser.add_argument("--dry-run", action="store_true", help="Call model and print result without writing files")
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing rule files")
    parser.add_argument("--print-prompt", action="store_true", help="Print the model prompt to stderr")
    args = parser.parse_args()

    input_path = _resolve_cli_path(args.input_path)
    dataset_path = _resolve_cli_path(args.dataset_path)
    manifest_path = _resolve_manifest(
        args.dataset_name,
        dataset_path,
        _resolve_cli_path(args.manifest_path),
    )
    if input_path is None and manifest_path is None:
        if dataset_path is not None and dataset_path.exists():
            input_path = dataset_path
        else:
            raise ValueError("请传入 --input-path，或传入 --manifest-path，或传入一个存在的 --dataset-path")

    dataset_name = _normalize_dataset_name(args.dataset_name, dataset_path, input_path, manifest_path)

    base_profiles = _load_base_profiles(Path(args.base_rule_file).resolve() if args.base_rule_file else None)
    base_profile = _choose_base_profile(base_profiles, args.base_profile)
    if manifest_path is not None and (input_path is None or input_path.name == "manifest.json"):
        entries = _load_entries(manifest_path)
        samples = _build_samples(
            manifest_path,
            entries,
            sample_count=max(1, args.sample_count),
            max_chars=max(400, args.max_chunk_chars),
        )
        source_description = {
            "mode": "manifest",
            "manifest_path": str(manifest_path),
            "base_dir": str(manifest_path.parent),
        }
        total_sources = len(entries)
        sample_titles = [str(item.get("title", "")).strip() for item in entries]
    else:
        assert input_path is not None
        samples, total_sources = _build_content_samples(
            input_path,
            sample_count=max(1, args.sample_count),
            max_chars=max(400, args.max_chunk_chars),
        )
        source_description = {
            "mode": "content",
            "input_path": str(input_path),
            "note": "Samples were read directly from text files without manifest metadata.",
        }
        sample_titles = [str(item.get("title", "")).strip() for item in samples]

    prompt = _build_prompt(
        dataset_name=dataset_name,
        source_description=source_description,
        sample_titles=sample_titles,
        total_sources=total_sources,
        base_profiles=base_profiles,
        base_profile=base_profile,
        samples=samples,
    )
    if args.print_prompt:
        _progress(prompt)

    _progress(f"dataset={dataset_name}; source={source_description}; sources={total_sources}; samples={len(samples)}")
    clients = OnlineClients(Settings.load(ROOT))
    generated: dict[str, Any] | None = None
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            generated = clients.chat_json(
                RULE_ANALYSIS_SYSTEM_PROMPT,
                prompt,
            )
            break
        except Exception as exc:
            if attempt < max_retries:
                _progress(f"LLM 调用失败 (attempt {attempt}/{max_retries}): {_truncate(str(exc), 200)}")
            else:
                raise
    assert generated is not None

    profile = _normalize_rule_profile(generated, base_profile)
    rule_map = _normalize_rule_map(dataset_name, generated)
    output_payload = {
        "dataset_name": dataset_name,
        "source": source_description,
        "analysis": generated.get("analysis", {}),
        "rule_profiles": {"profiles": {dataset_name: profile}},
        "rule_map": rule_map,
    }

    if args.dry_run:
        print(json.dumps(output_payload, ensure_ascii=False, indent=2))
        return

    output_dir = Path(args.output_root).resolve() / dataset_name
    rule_profiles_path = output_dir / "rule_profiles.json"
    rule_map_path = output_dir / "rule_map.json"
    analysis_path = output_dir / "rule_analysis.json"

    existing = [path for path in [rule_profiles_path, rule_map_path, analysis_path] if path.exists()]
    if existing and not args.force:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"规则文件已存在，使用 --force 覆盖: {existing_text}")

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(rule_profiles_path, output_payload["rule_profiles"])
    _write_json(rule_map_path, rule_map)
    _write_json(
        analysis_path,
        {
            "dataset_name": dataset_name,
            "source": source_description,
            "analysis": generated.get("analysis", {}),
            "samples": samples,
        },
    )

    print(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "rule_profiles_path": str(rule_profiles_path),
                "rule_map_path": str(rule_map_path),
                "analysis_path": str(analysis_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
