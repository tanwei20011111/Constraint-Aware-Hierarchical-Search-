from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from regtree_agent.config import Settings
    from regtree_agent.online import OnlineClients
    from regtree_agent.rules import (
        ExtractionRuleProfile,
        default_rule_file,
        default_rule_map_file,
        load_rule_map,
        load_rule_profiles,
        normalize_dataset_rule_name,
        resolve_rule_name_for_file,
    )
else:
    from .config import Settings
    from .online import OnlineClients
    from .rules import (
        ExtractionRuleProfile,
        default_rule_file,
        default_rule_map_file,
        load_rule_map,
        load_rule_profiles,
        normalize_dataset_rule_name,
        resolve_rule_name_for_file,
    )


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    title: str
    start_page: int
    end_page: int
    text: str
    source_path: str

    def searchable_text(self) -> str:
        return (
            f"文档:{self.document_id}\n标题:{self.title}\n页码:{self.start_page}-{self.end_page}\n"
            f"{self.text}"
        )


@dataclass(slots=True)
class InputEvidenceBundle:
    chunks: list[ChunkRecord]
    chapter_chunk_ids: list[str]
    heading_chunk_ids: dict[str, list[str]]
    subheading_chunk_ids: dict[tuple[str, str], list[str]]


@dataclass(slots=True)
class BuildRuleOptions:
    override_rule: str | None = None
    rule_map_path: Path | None = None
    rule_file: Path | None = None
    dataset_name: str | None = None
    dataset_path: Path | None = None
    llm_block_max_chars: int = 4000
    resume: bool = True
    checkpoint_path: Path | None = None
    print_llm_units: bool = False
    print_llm_window_split: bool = False
    print_llm_unit_attachments: bool = False


def _rule_storage_dir(settings: Settings, options: BuildRuleOptions | None = None) -> Path:
    if options is None:
        return settings.rag_storage_dir
    dataset_name = normalize_dataset_rule_name(options.dataset_name, options.dataset_path)
    if not dataset_name:
        return settings.rag_storage_dir
    return settings.rag_storage_dir / dataset_name


def _default_llm_checkpoint_file(settings: Settings, options: BuildRuleOptions | None = None) -> Path:
    return _rule_storage_dir(settings, options) / "regtree_llm_checkpoint.json"


def _llm_checkpoint_build_signature(options: BuildRuleOptions) -> dict[str, Any]:
    sig: dict[str, Any] = {
        "override_rule": options.override_rule,
        "rule_map_path": str(options.rule_map_path.resolve()) if options.rule_map_path else None,
        "rule_file": str(options.rule_file.resolve()) if options.rule_file else None,
        "dataset_name": options.dataset_name,
        "dataset_path": str(options.dataset_path.resolve()) if options.dataset_path else None,
        "llm_block_max_chars": int(options.llm_block_max_chars),
    }
    if options.rule_file and options.rule_file.exists():
        sig["rule_file_hash"] = hashlib.md5(options.rule_file.read_bytes()).hexdigest()[:12]
    return sig


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _save_llm_checkpoint(
    checkpoint_path: Path,
    *,
    options: BuildRuleOptions,
    input_files: list[str],
    root_id: str,
    nodes: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    completed_blocks: dict[str, int],
) -> None:
    _write_json_atomic(
        checkpoint_path,
        {
            "version": 1,
            "root_id": root_id,
            "build_signature": _llm_checkpoint_build_signature(options),
            "input_files": input_files,
            "completed_blocks": completed_blocks,
            "payload": {
                "root_id": root_id,
                "nodes": nodes,
                "chunks": chunks,
            },
        },
    )


def _load_llm_checkpoint(
    checkpoint_path: Path,
    *,
    options: BuildRuleOptions,
    input_files: list[str],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported LLM checkpoint version: {payload.get('version')}")
    if payload.get("build_signature") != _llm_checkpoint_build_signature(options):
        raise ValueError("LLM checkpoint 与当前构建参数不一致，请删除 checkpoint 后重试")
    if payload.get("input_files") != input_files:
        raise ValueError("LLM checkpoint 与当前 input 文件列表不一致，请删除 checkpoint 后重试")

    tree_payload = payload.get("payload") or {}
    root_id = str(payload.get("root_id", "root"))
    nodes = tree_payload.get("nodes") or {}
    chunks = tree_payload.get("chunks") or {}
    completed_blocks_raw = payload.get("completed_blocks") or {}
    completed_blocks = {str(name): int(count) for name, count in completed_blocks_raw.items()}
    return root_id, nodes, chunks, completed_blocks



def _parse_chunk_metadata_from_path(chunk_path: Path) -> tuple[str, int, int]:
    title = chunk_path.stem
    start_page = 0
    end_page = 0
    for line in chunk_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean.startswith("标题:"):
            title = clean.split(":", 1)[1].strip()
        elif clean.startswith("页码范围:"):
            page_text = clean.split(":", 1)[1].strip()
            match = re.match(r"(\d+)-(\d+)", page_text)
            if match:
                start_page = int(match.group(1))
                end_page = int(match.group(2))
    return title, start_page, end_page


def _split_text_into_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    return paragraphs


def _split_block_into_windows(
    text: str,
    *,
    max_chars: int,
    overlap_paragraphs: int,
) -> list[str]:
    paragraphs = _split_text_into_paragraphs(text)
    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        piece_len = len(paragraph) + (2 if current else 0)
        if current and current_len + piece_len > max_chars:
            windows.append("\n\n".join(current))
            keep = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current = list(keep)
            current_len = sum(len(item) for item in current) + max(0, 2 * (len(current) - 1))
        current.append(paragraph)
        current_len += piece_len
    if current:
        windows.append("\n\n".join(current))
    return windows or [text.strip()]


def _split_text_into_llm_blocks(
    text: str,
    *,
    max_chars: int,
) -> list[str]:
    paragraphs = _split_text_into_paragraphs(text)
    if not paragraphs:
        return [text.strip()]
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        piece_len = len(paragraph) + (2 if current else 0)
        if current and current_len + piece_len > max_chars:
            blocks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
            piece_len = len(paragraph)
        current.append(paragraph)
        current_len += piece_len
    if current:
        blocks.append("\n\n".join(current).strip())
    return blocks or [text.strip()]


def _build_llm_window_prompt(
    *,
    title: str,
    paragraphs: list[str],
    max_chars: int,
    overlap_paragraphs: int,
) -> dict[str, Any]:
    return {
        "task": (
            "请把法规文本按语义完整性切分为若干检索窗口。"
            "不要改写原文，不要生成新文本，只返回每个窗口覆盖的段落编号范围。"
            "尽量让定义、排除项、同一条款说明保持在同一窗口内。"
        ),
        "title": title,
        "constraints": {
            "max_chars_per_window": max_chars,
            "preferred_overlap_paragraphs": overlap_paragraphs,
            "paragraph_count": len(paragraphs),
            "indexing": "段落编号从 1 开始",
        },
        "paragraphs": [
            {
                "index": idx + 1,
                "chars": len(paragraph),
                "text": paragraph,
            }
            for idx, paragraph in enumerate(paragraphs)
        ],
        "output_schema": {
            "windows": [
                {
                    "start": "窗口起始段落编号（1-based, inclusive）",
                    "end": "窗口结束段落编号（1-based, inclusive）",
                    "reason": "一句中文说明为什么这样切",
                }
            ]
        },
    }


def _emit_llm_window_split_trace(
    *,
    title: str,
    source_path: str,
    prompt: dict[str, Any],
    payload: dict[str, Any],
    windows: list[str],
) -> None:
    trace = {
        "source_path": source_path,
        "title": title,
        "llm_window_prompt": prompt,
        "llm_window_payload": payload,
        "rebuilt_windows": windows,
    }
    print(
        "[tree_index] llm-window-split -> "
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}",
        file=sys.stderr,
        flush=True,
    )


def _llm_split_block_into_windows(
    *,
    clients: OnlineClients,
    title: str,
    source_path: str,
    text: str,
    max_chars: int,
    overlap_paragraphs: int,
    print_trace: bool = False,
) -> list[str]:
    paragraphs = _split_text_into_paragraphs(text)
    if not paragraphs:
        return [text.strip()]
    if len(paragraphs) == 1:
        return [paragraphs[0]]

    system_prompt = (
        "你是法规文本切分助手。"
        "你要根据语义完整性把文本划分成适合检索的窗口。"
        "不得改写原文，只能返回段落编号范围。"
        "输出必须是 JSON 对象，不要输出额外解释。"
    )
    prompt = _build_llm_window_prompt(
        title=title,
        paragraphs=paragraphs,
        max_chars=max_chars,
        overlap_paragraphs=overlap_paragraphs,
    )
    try:
        payload = clients.chat_json(system_prompt, json.dumps(prompt, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(
            "[tree_index] llm-window-split fallback -> "
            f"title={title!r}; source_path={source_path}; error={_summarize_exception(exc)}",
            file=sys.stderr,
            flush=True,
        )
        return _split_block_into_windows(
            text,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )
    raw_windows = payload.get("windows", [])
    if not isinstance(raw_windows, list) or not raw_windows:
        return _split_block_into_windows(
            text,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )

    merged_windows: list[str] = []
    paragraph_count = len(paragraphs)
    for item in raw_windows:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 1 or end < start or end > paragraph_count:
            continue
        window_text = "\n\n".join(paragraphs[start - 1 : end]).strip()
        if not window_text:
            continue
        if len(window_text) > max_chars * 2:
            # If the model returns an overlong semantic block, fall back to
            # deterministic local splitting to keep embedding windows bounded.
            merged_windows.extend(
                _split_block_into_windows(
                    window_text,
                    max_chars=max_chars,
                    overlap_paragraphs=overlap_paragraphs,
                )
            )
            continue
        merged_windows.append(window_text)

    if not merged_windows:
        return _split_block_into_windows(
            text,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )
    if print_trace:
        _emit_llm_window_split_trace(
            title=title,
            source_path=source_path,
            prompt=prompt,
            payload=payload,
            windows=merged_windows,
        )
    return merged_windows


def _build_input_evidence_chunks(
    *,
    document_id: str,
    source_chunk_id: str,
    title: str,
    start_page: int,
    end_page: int,
    text: str,
    source_path: str,
    clients: OnlineClients,
    window_max_chars: int = 1200,
    window_overlap_paragraphs: int = 1,
    print_llm_window_split: bool = False,
) -> InputEvidenceBundle:
    chunks: list[ChunkRecord] = []
    chapter_chunk_ids: list[str] = []
    heading_chunk_ids: dict[str, list[str]] = {}
    subheading_chunk_ids: dict[tuple[str, str], list[str]] = {}
    seq = 1

    def emit_chunk(local_title: str, local_text: str) -> str:
        nonlocal seq
        chunk_id = f"{document_id}::{source_chunk_id}::part{seq:03d}"
        seq += 1
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                title=local_title,
                start_page=start_page,
                end_page=end_page,
                text=local_text.strip(),
                source_path=source_path,
            )
        )
        chapter_chunk_ids.append(chunk_id)
        return chunk_id

    def split_windows(local_title: str, local_text: str) -> list[str]:
        return _llm_split_block_into_windows(
            clients=clients,
            title=local_title,
            source_path=source_path,
            text=local_text,
            max_chars=window_max_chars,
            overlap_paragraphs=window_overlap_paragraphs,
            print_trace=print_llm_window_split,
        )

    for index, window in enumerate(split_windows(title, text), start=1):
        emit_chunk(f"{title}-{index}", window)
    return InputEvidenceBundle(
        chunks=chunks,
        chapter_chunk_ids=chapter_chunk_ids,
        heading_chunk_ids=heading_chunk_ids,
        subheading_chunk_ids=subheading_chunk_ids,
    )


def _safe_read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_rule_assets(
    settings: Settings,
    options: BuildRuleOptions,
) -> tuple[Path, Path, str | None]:
    dataset_name = normalize_dataset_rule_name(options.dataset_name, options.dataset_path)
    resolved_rule_file = options.rule_file or default_rule_file(
        settings.workspace_root,
        dataset_name=dataset_name,
        dataset_path=options.dataset_path,
    )
    resolved_rule_map = options.rule_map_path or default_rule_map_file(
        settings.workspace_root,
        dataset_name=dataset_name,
        dataset_path=options.dataset_path,
    )
    return resolved_rule_file, resolved_rule_map, dataset_name


def _make_node(
    node_id: str,
    *,
    code: str,
    title: str,
    node_type: str,
    document_id: str,
    document_title: str,
    text: str,
    notes: list[str] | None = None,
    exclusions: list[str] | None = None,
    definitions: list[str] | None = None,
    evidence_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "code": code,
        "title": title,
        "node_type": node_type,
        "document_id": document_id,
        "document_title": document_title,
        "text": text.strip(),
        "notes": notes or [],
        "exclusions": exclusions or [],
        "definitions": definitions or [],
        "evidence_chunk_ids": evidence_chunk_ids or [],
        "children": [],
    }


def _normalize_code_digits(value: Any) -> str:
    return "".join(ch for ch in str(value).strip() if ch.isdigit())


def _normalize_generic_code(value: Any) -> str:
    text = str(value or "").strip()
    text = text.strip("：:，,；;。")
    text = re.sub(r"\s+", "", text)
    return text[:80]


def _validate_node_code(code: str, node_type: str) -> str:
    """Reject obviously broken codes (single char, pure punctuation, etc.) without
    assuming any specific coding scheme like HS 4/6-digit."""
    if not code:
        return code
    clean = code.strip("_-. ")
    if len(clean) <= 1:
        return ""
    if all(not ch.isalnum() for ch in clean):
        return ""
    return code


def _coerce_str_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _cosine_similarity_matrix(query_vectors: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
    if query_vectors.size == 0 or candidate_vectors.size == 0:
        return np.zeros((len(query_vectors), len(candidate_vectors)), dtype=np.float32)
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    query_norms = np.clip(query_norms, 1e-12, None)
    candidate_norms = np.clip(candidate_norms, 1e-12, None)
    normalized_queries = query_vectors / query_norms
    normalized_candidates = candidate_vectors / candidate_norms
    return normalized_queries @ normalized_candidates.T


def _build_structure_attachment_text(
    *,
    node_type: str,
    title: str,
    code: str = "",
    parent_title: str = "",
    parent_code: str = "",
    text: str = "",
    notes: list[str] | None = None,
    exclusions: list[str] | None = None,
    definitions: list[str] | None = None,
) -> str:
    parts = [
        f"节点类型:{node_type}",
        f"编码:{code}",
        f"标题:{title}",
        f"上级编码:{parent_code}",
        f"上级标题:{parent_title}",
        text.strip(),
        "\n".join(definitions or []),
        "\n".join(exclusions or []),
        "\n".join(notes or []),
    ]
    return "\n".join(part for part in parts if part and part.strip())


def _coerce_node_type(value: Any, *, has_children: bool) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if text:
        return text
    return "section" if has_children else "reference"


def _payload_unit_to_generic_unit(unit: dict[str, Any]) -> dict[str, Any]:
    children = unit.get("children", []) if isinstance(unit.get("children"), list) else []
    return {
        "node_type": _coerce_node_type(unit.get("node_type", ""), has_children=bool(children)),
        "code": _normalize_generic_code(unit.get("code", "")),
        "title": str(unit.get("title", "")).strip().rstrip("：:"),
        "text": str(unit.get("text", "")).strip(),
        "notes": _coerce_str_list(unit.get("notes"), limit=6),
        "exclusions": _coerce_str_list(unit.get("exclusions"), limit=6),
        "definitions": _coerce_str_list(unit.get("definitions"), limit=6),
        "children": [
            _payload_unit_to_generic_unit(child)
            for child in children
            if isinstance(child, dict)
        ],
    }


def _legacy_payload_to_generic_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("units"), list):
        return [
            _payload_unit_to_generic_unit(unit)
            for unit in payload["units"]
            if isinstance(unit, dict)
        ]

    units: list[dict[str, Any]] = []
    if bool(payload.get("is_chapter", False)) and isinstance(payload.get("chapter"), dict):
        chapter = payload["chapter"]
        heading_units: list[dict[str, Any]] = []
        for heading in chapter.get("headings", []) if isinstance(chapter.get("headings"), list) else []:
            if not isinstance(heading, dict):
                continue
            sub_units: list[dict[str, Any]] = []
            for subheading in heading.get("subheadings", []) if isinstance(heading.get("subheadings"), list) else []:
                if not isinstance(subheading, dict):
                    continue
                suffix = _normalize_code_digits(subheading.get("suffix", ""))
                title = str(subheading.get("title", "")).strip().rstrip("：:")
                if len(suffix) != 2 or not title:
                    continue
                parent_code = _normalize_code_digits(heading.get("code", ""))
                sub_units.append(
                    {
                        "node_type": "subheading",
                        "code": f"{parent_code}{suffix}",
                        "title": title,
                        "text": str(subheading.get("text", "")).strip(),
                        "notes": _coerce_str_list(subheading.get("notes"), limit=6),
                        "exclusions": _coerce_str_list(subheading.get("exclusions"), limit=6),
                        "definitions": _coerce_str_list(subheading.get("definitions"), limit=6),
                        "children": [],
                    }
                )
            heading_units.append(
                {
                    "node_type": "heading",
                    "code": _normalize_code_digits(heading.get("code", "")),
                    "title": str(heading.get("title", "")).strip().rstrip("：:"),
                    "text": str(heading.get("text", "")).strip(),
                    "notes": _coerce_str_list(heading.get("notes"), limit=6),
                    "exclusions": _coerce_str_list(heading.get("exclusions"), limit=6),
                    "definitions": _coerce_str_list(heading.get("definitions"), limit=6),
                    "children": sub_units,
                }
            )
        units.append(
            {
                "node_type": "chapter",
                "code": _normalize_code_digits(chapter.get("code", "")),
                "title": str(chapter.get("title", "")).strip().rstrip("：:"),
                "text": "",
                "notes": _coerce_str_list(chapter.get("notes"), limit=6),
                "exclusions": _coerce_str_list(chapter.get("exclusions"), limit=6),
                "definitions": _coerce_str_list(chapter.get("definitions"), limit=6),
                "children": heading_units,
            }
        )
        return units

    if isinstance(payload.get("reference"), dict):
        reference = payload["reference"]
        units.append(
            {
                "node_type": "reference",
                "code": "",
                "title": str(reference.get("title", "")).strip().rstrip("：:"),
                "text": "",
                "notes": _coerce_str_list(reference.get("notes"), limit=6),
                "exclusions": _coerce_str_list(reference.get("exclusions"), limit=6),
                "definitions": _coerce_str_list(reference.get("definitions"), limit=6),
                "children": [],
            }
        )
    return units


def _normalize_llm_units(
    payload: dict[str, Any],
    *,
    fallback_title: str,
    fallback_text: str,
    fallback_notes: list[str],
    fallback_exclusions: list[str],
    fallback_definitions: list[str],
) -> list[dict[str, Any]]:
    units = _legacy_payload_to_generic_units(payload)
    normalized = [unit for unit in units if unit.get("title") or unit.get("text") or unit.get("children")]
    if normalized:
        return normalized
    return [
        {
            "node_type": "reference",
            "code": "",
            "title": fallback_title,
            "text": fallback_text,
            "notes": fallback_notes,
            "exclusions": fallback_exclusions,
            "definitions": fallback_definitions,
            "children": [],
        }
    ]


def _flatten_generic_units_for_matching(
    units: list[dict[str, Any]],
    *,
    parent_title: str = "",
    parent_code: str = "",
    path_prefix: tuple[int, ...] = (),
) -> list[tuple[tuple[int, ...], dict[str, Any], str, str]]:
    entries: list[tuple[tuple[int, ...], dict[str, Any], str, str]] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            continue
        path = path_prefix + (index,)
        entries.append((path, unit, parent_title, parent_code))
        entries.extend(
            _flatten_generic_units_for_matching(
                unit.get("children", []) if isinstance(unit.get("children"), list) else [],
                parent_title=str(unit.get("title", "")).strip(),
                parent_code=_normalize_generic_code(unit.get("code", "")),
                path_prefix=path,
            )
        )
    return entries


def _match_generic_units_to_chunks(
    *,
    clients: OnlineClients,
    evidence_bundle: InputEvidenceBundle,
    units: list[dict[str, Any]],
    return_trace: bool = False,
) -> dict[tuple[int, ...], list[str]] | tuple[dict[tuple[int, ...], list[str]], list[dict[str, Any]]]:
    chunk_records = evidence_bundle.chunks
    if not chunk_records or not units:
        return {}

    flattened = _flatten_generic_units_for_matching(units)
    if not flattened:
        return {}

    query_texts = [
        _build_structure_attachment_text(
            node_type=str(unit.get("node_type", "")).strip(),
            title=str(unit.get("title", "")).strip(),
            code=_normalize_generic_code(unit.get("code", "")),
            parent_title=parent_title,
            parent_code=parent_code,
            text=str(unit.get("text", "")).strip(),
            notes=unit.get("notes", []),
            exclusions=unit.get("exclusions", []),
            definitions=unit.get("definitions", []),
        )
        for _, unit, parent_title, parent_code in flattened
    ]
    chunk_texts = [
        f"标题:{chunk.title}\n文档:{chunk.document_id}\n{chunk.text}"
        for chunk in chunk_records
    ]
    query_vectors = np.asarray(clients.embed_texts(query_texts, batch_size=8), dtype=np.float32)
    chunk_vectors = np.asarray(clients.embed_texts(chunk_texts, batch_size=8), dtype=np.float32)
    similarity = _cosine_similarity_matrix(query_vectors, chunk_vectors)

    matches: dict[tuple[int, ...], list[str]] = {}
    trace_rows: list[dict[str, Any]] = []
    for row_index, (path, unit, _, _) in enumerate(flattened):
        row = similarity[row_index]
        if row.size == 0:
            continue
        children = unit.get("children", []) if isinstance(unit.get("children"), list) else []
        top_k = 2 if children else 1
        ranked = np.argsort(row)[::-1][:top_k]
        matched_ids = [chunk_records[idx].chunk_id for idx in ranked if row[idx] > 0.15]
        if not matched_ids:
            matched_ids = []
        matches[path] = matched_ids
        if return_trace:
            trace_rows.append(
                {
                    "path": list(path),
                    "node_type": str(unit.get("node_type", "")).strip(),
                    "code": _normalize_generic_code(unit.get("code", "")),
                    "title": str(unit.get("title", "")).strip(),
                    "selected_evidence_chunk_ids": matched_ids,
                    "matched_chunks": [
                        {
                            "chunk_id": chunk_records[idx].chunk_id,
                            "chunk_title": chunk_records[idx].title,
                            "score": float(row[idx]),
                        }
                        for idx in ranked
                    ],
                }
            )
    # Evidence 层级传播：子节点继承父节点的 evidence chunk，确保细粒度节点
    # 也能获得上层相关上下文。
    _propagate_evidence_to_children(matches)

    if return_trace:
        return matches, trace_rows
    return matches


def _propagate_evidence_to_children(
    matches: dict[tuple[int, ...], list[str]],
    *,
    max_inherited: int = 2,
) -> None:
    """按层级拓扑顺序，把父节点的 evidence chunk 传播给子节点。

    子节点保留自己的高相关 chunk 在前，父节点的 chunk 追加在后作为补充上下文。
    """
    if not matches:
        return
    # 按 path 长度排序，确保父节点先于子节点处理
    sorted_paths = sorted(matches.keys(), key=lambda p: (len(p), p))
    for path in sorted_paths:
        if len(path) <= 1:
            continue
        parent_path = path[:-1]
        parent_chunks = matches.get(parent_path, [])
        if not parent_chunks:
            continue
        own_chunks = matches[path]
        # 去重合并：自己的 chunk 优先，父节点的作为补充
        combined: list[str] = []
        seen: set[str] = set()
        for cid in own_chunks:
            if cid not in seen:
                seen.add(cid)
                combined.append(cid)
        for cid in parent_chunks:
            if cid not in seen:
                seen.add(cid)
                combined.append(cid)
                if len(combined) >= len(own_chunks) + max_inherited:
                    break
        matches[path] = combined


def _build_llm_chunk_prompt(
    *,
    title: str,
    text: str,
    hierarchy_hints: str = "",
    code_format_hint: str = "",
) -> dict[str, Any]:
    base_task = (
        "把当前文本块抽取为通用层级结构。"
        "不要假设文档一定具有固定的章、节、品目或子目格式。"
        "请只依据原文中可识别的层级、标题、编号、说明、排除项和定义来构造节点树。"
        "关键要求："
        "1. 排除项(exclusions)必须完整保留原文——包括'不包括'、'除外'、'不归入'等表达后的所有文字。"
        "2. 定义(definitions)必须保留原文——包括'包括'、'所称'、'是指'等表达后的所有文字。"
        "3. text 字段应直接引用原文，不要摘要缩写，保留足够上下文。"
        "4. 层级深度由原文决定，不要固定为三层。"
        "5. 如果某个子项下面继续出现数字、字母、括号编号、罗马数字或其他下级编号，必须作为 children 继续嵌套。"
        "6. 不要把下级编号扁平化到同一层。"
        "7. 如果文本块内部没有清晰层级，也可以只返回一个单节点。"
    )
    hierarchy_hints_stripped = hierarchy_hints.strip() if hierarchy_hints else ""
    if hierarchy_hints_stripped:
        base_task += f"8. 本文档的层级编码规则如下，必须严格遵循：{hierarchy_hints_stripped}"
    code_desc = (
        code_format_hint.strip()
        if code_format_hint and code_format_hint.strip()
        else "节点编号；保留原文编号或可追溯的层级编号，例如 a、1、a.1；若无编号可为空字符串"
    )
    prompt: dict[str, Any] = {
        "task": base_task,
        "title": title,
        "text": text,
        "output_schema": {
            "units": [
                {
                    "node_type": "节点类型，例如 chapter、section、heading、subheading、reference、entry；若难判断可写 section 或 reference",
                    "code": code_desc,
                    "title": "节点标题；若无显式标题可给简短概括",
                    "text": "该节点最能代表其语义范围的原文摘要性摘录，尽量直接引用原文，不要编造",
                    "notes": "一般说明数组",
                    "exclusions": "排除项数组",
                    "definitions": "定义性内容数组",
                    "children": "同结构的子节点数组；若无子节点则为空数组",
                }
            ]
        },
    }
    return prompt


def _llm_extract_chunk_structure(
    *,
    clients: OnlineClients,
    title: str,
    text: str,
    hierarchy_hints: str = "",
    code_format_hint: str = "",
) -> dict[str, Any]:
    system_prompt = (
        "你是法规文本结构化抽取助手。"
        "你需要从法规原文中抽取层级结构、说明、排除项和定义信息。"
        "不要编造原文中不存在的编码或标题。"
        "输出必须是 JSON 对象，不要输出额外解释。"
    )
    prompt = _build_llm_chunk_prompt(title=title, text=text, hierarchy_hints=hierarchy_hints, code_format_hint=code_format_hint)
    return clients.chat_json(system_prompt, json.dumps(prompt, ensure_ascii=False, indent=2))


def _emit_llm_units_trace(
    *,
    source_path: str,
    title: str,
    payload: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    trace = {
        "source_path": source_path,
        "title": title,
        "llm_payload": payload,
        "normalized_units": units,
    }
    print(
        "[tree_index] llm-units -> "
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}",
        file=sys.stderr,
        flush=True,
    )


def _summarize_exception(exc: Exception, *, limit: int = 240) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"\s+", " ", message).strip()
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _emit_llm_unit_attachments_trace(
    *,
    source_path: str,
    title: str,
    attachments: list[dict[str, Any]],
) -> None:
    trace = {
        "source_path": source_path,
        "title": title,
        "unit_attachments": attachments,
    }
    print(
        "[tree_index] llm-unit-attachments -> "
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}",
        file=sys.stderr,
        flush=True,
    )


def _build_generic_node_id(
    *,
    parent_id: str,
    unit: dict[str, Any],
    sibling_index: int,
    existing_ids: set[str],
) -> str:
    base = _normalize_generic_code(unit.get("code", ""))
    base = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", base).strip("_")
    if not base:
        raw_title = str(unit.get("title", "")).strip().lower()
        slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", raw_title).strip("_")
        base = slug[:32] if slug else f"u{sibling_index:02d}"
    candidate = f"{parent_id}::{base}"
    if candidate not in existing_ids:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def _format_unit_progress_label(unit: dict[str, Any]) -> str:
    node_type = str(unit.get("node_type", "")).strip() or "unit"
    code = _normalize_generic_code(unit.get("code", ""))
    title = str(unit.get("title", "")).strip() or "(untitled)"
    prefix = f"{node_type}:{code}" if code else node_type
    return f"{prefix} {title}"


def _materialize_generic_units(
    *,
    nodes: dict[str, dict[str, Any]],
    parent_id: str,
    units: list[dict[str, Any]],
    document_id: str,
    document_title: str,
    default_text: str,
    default_chunk_ids: list[str],
    matched_chunk_ids: dict[tuple[int, ...], list[str]],
    rule_name: str | None = None,
    path_prefix: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    progress_prefix: str = "",
) -> None:
    for sibling_index, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            continue
        path = path_prefix + (sibling_index - 1,)
        children = unit.get("children", []) if isinstance(unit.get("children"), list) else []
        node_id = _build_generic_node_id(
            parent_id=parent_id,
            unit=unit,
            sibling_index=sibling_index,
            existing_ids=set(nodes.keys()),
        )
        raw_code = _normalize_generic_code(unit.get("code", ""))
        node_type = _coerce_node_type(unit.get("node_type", ""), has_children=bool(children))
        validated_code = _validate_node_code(raw_code, node_type)
        node = _make_node(
            node_id,
            code=validated_code,
            title=str(unit.get("title", "")).strip() or f"unit_{sibling_index}",
            node_type=node_type,
            document_id=document_id,
            document_title=document_title,
            text=str(unit.get("text", "")).strip() or default_text,
            notes=unit.get("notes", []),
            exclusions=unit.get("exclusions", []),
            definitions=unit.get("definitions", []),
            evidence_chunk_ids=matched_chunk_ids.get(path, default_chunk_ids),
        )
        if rule_name is not None:
            node["rule_name"] = rule_name
        nodes[node_id] = node
        _append_child(nodes, parent_id, node_id)
        if progress is not None:
            progress(
                f"{progress_prefix}识别节点 -> {_format_unit_progress_label(unit)} "
                f"(evidence={len(node.get('evidence_chunk_ids', []))})"
            )
        _materialize_generic_units(
            nodes=nodes,
            parent_id=node_id,
            units=children,
            document_id=document_id,
            document_title=document_title,
            default_text=default_text,
            default_chunk_ids=matched_chunk_ids.get(path, default_chunk_ids),
            matched_chunk_ids=matched_chunk_ids,
            rule_name=rule_name,
            path_prefix=path,
            progress=progress,
            progress_prefix=progress_prefix,
        )


def _build_input_document_with_llm(
    settings: Settings,
    rule_options: BuildRuleOptions,
    clients: OnlineClients,
    nodes: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    root_id: str,
    document_id: str = "input",
    document_title: str = "input",
    completed_blocks: dict[str, int] | None = None,
    checkpoint_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    resolved_rule_file, resolved_rule_map, dataset_name = _resolve_rule_assets(settings, rule_options)
    profiles = load_rule_profiles(str(resolved_rule_file))
    rule_map = load_rule_map(resolved_rule_map)
    completed_blocks = completed_blocks or {}
    document_node_id = f"doc::{document_id}"
    document_node = nodes.get(document_node_id)
    if document_node is None:
        document_node = _make_node(
            document_node_id,
            code="",
            title=document_title,
            node_type="document",
            document_id=document_id,
            document_title=document_title,
            text=document_title,
        )
        nodes[document_node["id"]] = document_node
    if document_node["id"] not in nodes[root_id]["children"]:
        _append_child(nodes, root_id, document_node["id"])

    input_files = sorted(settings.input_dir.glob("*.txt"))
    input_file_names = [item.name for item in input_files]
    for index, chunk_path in enumerate(input_files, start=1):
        rule_name = resolve_rule_name_for_file(
            chunk_path.name,
            override_rule=rule_options.override_rule,
            rule_map=rule_map,
        )
        rule = profiles.get(rule_name)
        title, start_page, end_page = _parse_chunk_metadata_from_path(chunk_path)
        source_chunk_id = chunk_path.stem
        chunk_text = chunk_path.read_text(encoding="utf-8")
        block_max_chars = max(rule_options.llm_block_max_chars, rule.window_max_chars if rule else 1200)
        if progress is not None:
            progress(
                f"LLM建树: 文件 {index}/{len(input_files)} -> {chunk_path.name} "
                f"(title={title}, chars={len(chunk_text)}, dataset={dataset_name or 'default'})"
            )
        llm_blocks = _split_text_into_llm_blocks(
            chunk_text,
            max_chars=block_max_chars,
        )
        if progress is not None:
            progress(
                f"LLM建树: {chunk_path.name} 粗分块完成 -> blocks={len(llm_blocks)}, "
                f"block_max_chars={max(rule_options.llm_block_max_chars, rule.window_max_chars)}"
            )
        completed_count = completed_blocks.get(chunk_path.name, 0)
        if completed_count > len(llm_blocks):
            raise ValueError(
                f"LLM checkpoint 记录的已完成 block 数超过当前实际 block 数: "
                f"{chunk_path.name} ({completed_count}>{len(llm_blocks)})"
            )
        if completed_count and progress is not None:
            progress(
                f"LLM建树: {chunk_path.name} 从 checkpoint 恢复 -> "
                f"跳过 {completed_count}/{len(llm_blocks)} 个 block"
            )
        for block_index, block_text in enumerate(llm_blocks, start=1):
            if block_index <= completed_count:
                continue
            block_suffix = "" if len(llm_blocks) == 1 else f" [block {block_index}]"
            block_title = f"{title}{block_suffix}"
            block_source_chunk_id = (
                source_chunk_id if len(llm_blocks) == 1 else f"{source_chunk_id}_seg{block_index:02d}"
            )
            if progress is not None:
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                    f"切窗开始 (chars={len(block_text)})"
                )
            evidence_bundle = _build_input_evidence_chunks(
                document_id=document_id,
                source_chunk_id=block_source_chunk_id,
                title=block_title,
                start_page=start_page,
                end_page=end_page,
                text=block_text,
                source_path=str(chunk_path),
                clients=clients,
                window_max_chars=rule.window_max_chars if rule else 1200,
                window_overlap_paragraphs=rule.window_overlap_paragraphs if rule else 1,
                print_llm_window_split=rule_options.print_llm_window_split,
            )
            if progress is not None:
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                    f"切窗完成 (windows={len(evidence_bundle.chunks)})"
                )
            for evidence_chunk in evidence_bundle.chunks:
                chunks[evidence_chunk.chunk_id] = {
                    "chunk_id": evidence_chunk.chunk_id,
                    "document_id": evidence_chunk.document_id,
                    "title": evidence_chunk.title,
                    "start_page": evidence_chunk.start_page,
                    "end_page": evidence_chunk.end_page,
                    "text": evidence_chunk.text,
                    "source_path": evidence_chunk.source_path,
                    "rule_name": rule_name,
                }

            fallback_notes, fallback_exclusions, fallback_definitions = _summarize_chunk_text(block_text)
            if progress is not None:
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> 结构抽取开始"
                )
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                    f"等待模型返回结构化节点"
                )
            max_retries = 6
            payload: dict[str, Any] = {}
            for attempt in range(1, max_retries + 1):
                try:
                    payload = _llm_extract_chunk_structure(
                        clients=clients,
                        title=block_title,
                        text=block_text,
                        hierarchy_hints=rule.hierarchy_hints if rule else "",
                        code_format_hint=rule.code_format_hint if rule else "",
                    )
                    break
                except Exception as exc:
                    is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower() or "limit" in str(exc).lower()
                    if is_rate_limit and attempt < max_retries:
                        wait_seconds = min(30, 5 * (2 ** (attempt - 1)))
                        if progress is not None:
                            progress(
                                f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                                f"触发限流，第 {attempt}/{max_retries} 次重试，等待 {wait_seconds}s..."
                            )
                        import time
                        time.sleep(wait_seconds)
                    elif attempt < max_retries:
                        if progress is not None:
                            progress(
                                f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                                f"结构抽取失败 ({_summarize_exception(exc)})，第 {attempt}/{max_retries} 次重试..."
                            )
                    else:
                        if progress is not None:
                            progress(
                                f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                                f"结构抽取失败，已重试 {max_retries} 次，终止 ({_summarize_exception(exc)})"
                            )
                        raise
            units = _normalize_llm_units(
                payload,
                fallback_title=block_title,
                fallback_text=block_text,
                fallback_notes=fallback_notes,
                fallback_exclusions=fallback_exclusions,
                fallback_definitions=fallback_definitions,
            )
            if progress is not None:
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                    f"结构抽取完成 (units={len(units)})"
                )
            if rule_options.print_llm_units:
                _emit_llm_units_trace(
                    source_path=str(chunk_path),
                    title=block_title,
                    payload=payload,
                    units=units,
                )
            if rule_options.print_llm_unit_attachments:
                matched_chunk_ids, attachment_trace = _match_generic_units_to_chunks(
                    clients=clients,
                    evidence_bundle=evidence_bundle,
                    units=units,
                    return_trace=True,
                )
                _emit_llm_unit_attachments_trace(
                    source_path=str(chunk_path),
                    title=block_title,
                    attachments=attachment_trace,
                )
            else:
                if progress is not None:
                    progress(
                        f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> 证据挂接开始"
                    )
                matched_chunk_ids = _match_generic_units_to_chunks(
                    clients=clients,
                    evidence_bundle=evidence_bundle,
                    units=units,
                )
            if progress is not None:
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                    f"证据挂接完成 (matched_units={len(matched_chunk_ids)})"
                )
            _materialize_generic_units(
                nodes=nodes,
                parent_id=document_node["id"],
                units=units,
                document_id=document_id,
                document_title=document_title,
                default_text=block_text,
                default_chunk_ids=evidence_bundle.chapter_chunk_ids,
                matched_chunk_ids=matched_chunk_ids,
                rule_name=rule_name,
                progress=progress,
                progress_prefix=(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> "
                ),
            )
            if progress is not None:
                progress(
                    f"LLM建树: {chunk_path.name} block {block_index}/{len(llm_blocks)} -> 节点写入完成"
                )
            completed_blocks[chunk_path.name] = block_index
            if checkpoint_path is not None:
                _save_llm_checkpoint(
                    checkpoint_path,
                    options=rule_options,
                    input_files=input_file_names,
                    root_id=root_id,
                    nodes=nodes,
                    chunks=chunks,
                    completed_blocks=completed_blocks,
                )


def _summarize_chunk_text(
    text: str,
    limit: int = 14,
) -> tuple[list[str], list[str], list[str]]:
    notes: list[str] = []
    exclusions: list[str] = []
    definitions: list[str] = []
    seen = 0
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if "排除" in clean or "不包括" in clean:
            for sentence in re.split(r"[。；]", clean):
                sentence = sentence.strip()
                if sentence and ("排除" in sentence or "不包括" in sentence):
                    exclusions.append(sentence)
        elif "定义" in clean or "是指" in clean or "包括" in clean:
            definitions.append(clean)
        else:
            notes.append(clean)
        seen += 1
        if seen >= limit:
            break
    return notes[:6], exclusions[:6], definitions[:6]


def _append_child(nodes: dict[str, dict[str, Any]], parent_id: str, child_id: str) -> None:
    if child_id not in nodes[parent_id]["children"]:
        nodes[parent_id]["children"].append(child_id)


def build_tree_payload(
    settings: Settings,
    rule_options: BuildRuleOptions | None = None,
    clients: OnlineClients | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """使用 LLM 从 input_dir 的 txt 文件构建法规树。"""
    options = rule_options or BuildRuleOptions()
    nodes: dict[str, dict[str, Any]] = {}
    chunks: dict[str, dict[str, Any]] = {}
    root_id = "root"
    nodes[root_id] = _make_node(
        root_id,
        code="",
        title="法规树根节点",
        node_type="root",
        document_id="all",
        document_title="all",
        text="法规树根节点",
    )

    input_files = sorted(settings.input_dir.glob("*.txt"))
    if not input_files:
        raise ValueError(f"input_dir 中没有找到 txt 文件: {settings.input_dir}")
    if clients is None:
        raise ValueError("LLM 建树需要 OnlineClients")

    checkpoint_path = options.checkpoint_path or _default_llm_checkpoint_file(settings, options)
    completed_blocks: dict[str, int] = {}
    input_file_names = [item.name for item in input_files]
    if options.resume:
        if checkpoint_path.exists():
            root_id, nodes, chunks, completed_blocks = _load_llm_checkpoint(
                checkpoint_path,
                options=options,
                input_files=input_file_names,
            )
            if progress is not None:
                progress(
                    f"LLM建树: 已恢复 checkpoint -> {checkpoint_path} "
                    f"(nodes={len(nodes)}, chunks={len(chunks)})"
                )
        elif progress is not None:
            progress(f"LLM建树: 未找到 checkpoint，开始全量构建 -> {checkpoint_path}")
    elif checkpoint_path.exists():
        checkpoint_path.unlink()
        if progress is not None:
            progress(f"LLM建树: 检测到旧 checkpoint，已清理 -> {checkpoint_path}")

    if root_id not in nodes:
        nodes[root_id] = _make_node(
            root_id,
            code="",
            title="法规树根节点",
            node_type="root",
            document_id="all",
            document_title="all",
            text="法规树根节点",
        )

    _build_input_document_with_llm(
        settings,
        options,
        clients,
        nodes,
        chunks,
        root_id,
        completed_blocks=completed_blocks,
        checkpoint_path=checkpoint_path,
        progress=progress,
    )

    # P2: 后处理去重，合并 code+title 完全相同的节点
    if nodes:
        dedup_count = _deduplicate_nodes(nodes)
        if progress is not None and dedup_count > 0:
            progress(f"树去重: 合并 {dedup_count} 个重复节点")

    return {"root_id": root_id, "nodes": nodes, "chunks": chunks}


def _deduplicate_nodes(nodes: dict[str, dict[str, Any]]) -> int:
    """合并 code+title 完全相同的节点，返回合并的节点数。

    保留首次出现的节点，将重复节点的 children 和 evidence 合并到保留节点，
    并更新所有父节点的 children 引用。
    """
    key_to_keep_id: dict[tuple[str, str], str] = {}
    duplicates: dict[str, str] = {}  # dup_id -> keep_id

    for node_id, node in list(nodes.items()):
        code = str(node.get("code", "")).strip()
        title = str(node.get("title", "")).strip()
        if not code or not title:
            continue
        key = (code, title)
        if key in key_to_keep_id:
            duplicates[node_id] = key_to_keep_id[key]
        else:
            key_to_keep_id[key] = node_id

    if not duplicates:
        return 0

    # 合并重复节点的 children 和 evidence
    for dup_id, keep_id in duplicates.items():
        dup_node = nodes.get(dup_id)
        keep_node = nodes.get(keep_id)
        if dup_node is None or keep_node is None:
            continue
        # children
        for child_id in dup_node.get("children", []):
            if child_id not in keep_node["children"]:
                keep_node["children"].append(child_id)
        # evidence_chunk_ids（去重）
        seen_evidence = set(keep_node.get("evidence_chunk_ids", []))
        for cid in dup_node.get("evidence_chunk_ids", []):
            if cid not in seen_evidence:
                seen_evidence.add(cid)
                keep_node.setdefault("evidence_chunk_ids", []).append(cid)
        # text 合并（若不同且非空）
        dup_text = str(dup_node.get("text", "")).strip()
        keep_text = str(keep_node.get("text", "")).strip()
        if dup_text and dup_text != keep_text:
            if keep_text:
                keep_node["text"] = keep_text + "\n" + dup_text
            else:
                keep_node["text"] = dup_text

    # 更新所有父节点的 children 引用（把指向 dup_id 的改为 keep_id）
    for node in nodes.values():
        new_children: list[str] = []
        seen: set[str] = set()
        for child_id in node.get("children", []):
            real_id = duplicates.get(child_id, child_id)
            if real_id not in seen and real_id in nodes:
                seen.add(real_id)
                new_children.append(real_id)
        node["children"] = new_children

    # 删除重复节点
    for dup_id in list(duplicates.keys()):
        nodes.pop(dup_id, None)

    return len(duplicates)


def _node_search_text(
    node: dict[str, Any],
    parent_node: dict[str, Any] | None = None,
) -> str:
    parts = [
        f"类型:{node['node_type']}",
        f"编码:{node['code']}",
        f"标题:{node['title']}",
    ]
    if parent_node:
        parts.append(f"上级:{parent_node['code']} {parent_node['title']}")
    parts.append(f"文档:{node['document_title']}")
    parts.append(node["text"].strip())
    if node["definitions"]:
        parts.append("定义:" + "；".join(node["definitions"][:3]))
    if node["exclusions"]:
        parts.append("排除:" + "；".join(node["exclusions"][:3]))
    if node["notes"]:
        parts.append("注释:" + "；".join(node["notes"][:3]))
    return "\n".join(part for part in parts if part and part.strip())


def _validate_tree(payload: dict[str, Any], *, strict: bool = False) -> list[str]:
    issues: list[str] = []
    nodes = payload["nodes"]
    chunks = payload["chunks"]
    all_reachable: set[str] = set()

    def walk(node_id: str) -> None:
        all_reachable.add(node_id)
        node = nodes.get(node_id)
        if node is None:
            issues.append(f"broken reference: {node_id}")
            return
        if not node.get("title"):
            issues.append(f"missing title: {node_id}")
        for child_id in node.get("children", []):
            walk(child_id)

    walk("root")
    orphaned = set(nodes.keys()) - all_reachable
    if orphaned:
        issues.append(f"orphaned nodes ({len(orphaned)}): {sorted(orphaned)[:10]}")

    for node_id, node in nodes.items():
        for chunk_id in node.get("evidence_chunk_ids", []):
            if chunk_id not in chunks:
                issues.append(f"broken evidence: {node_id} -> {chunk_id}")

    if strict:
        for node_id, node in nodes.items():
            if node["node_type"] == "heading":
                digits = "".join(ch for ch in node["code"] if ch.isdigit())
                if len(digits) != 4:
                    issues.append(f"suspect heading code: {node_id} code={node['code']}")
    return issues


def _build_vectors_for_payload(
    payload: dict[str, Any],
    clients: OnlineClients,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    node_ids = list(payload["nodes"].keys())
    chunk_ids = list(payload["chunks"].keys())
    parent_by_id: dict[str, str] = {}
    for nid, n in payload["nodes"].items():
        for child_id in n.get("children", []):
            parent_by_id[str(child_id)] = str(nid)
    node_texts = []
    for node_id in node_ids:
        node = payload["nodes"][node_id]
        pid = parent_by_id.get(node_id)
        parent_node = payload["nodes"].get(pid) if pid else None
        node_texts.append(_node_search_text(node, parent_node))

    chunk_to_nodes: dict[str, list[dict[str, Any]]] = {}
    for n in payload["nodes"].values():
        for cid in n.get("evidence_chunk_ids", []):
            chunk_to_nodes.setdefault(cid, []).append(n)
    chunk_texts = []
    for chunk_id in chunk_ids:
        chunk = payload["chunks"][chunk_id]
        parent_nodes = chunk_to_nodes.get(chunk_id, [])
        parent_context = ""
        if parent_nodes:
            pn = parent_nodes[0]
            parent_context = f"所属:{pn['code']} {pn['title']}\n"
        chunk_texts.append(
            f"标题:{chunk['title']}\n{parent_context}文档:{chunk['document_id']}\n{chunk['text']}"
        )

    all_texts = node_texts + chunk_texts
    total_texts = len(all_texts)
    if progress is not None:
        progress(
            f"开始生成向量: total={total_texts} "
            f"(nodes={len(node_texts)}, chunks={len(chunk_texts)})"
        )
    all_vectors = np.asarray(
        clients.embed_texts(all_texts, progress=progress, label="embedding"),
        dtype=np.float32,
    )
    node_count = len(node_texts)
    node_vectors = all_vectors[:node_count]
    chunk_vectors = all_vectors[node_count:]
    return (
        node_ids,
        chunk_ids,
        node_vectors,
        chunk_vectors,
    )


def rebuild_vectors_from_tree(
    clients: OnlineClients,
    *,
    tree_path: Path,
    vectors_path: Path,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    if progress is not None:
        progress(f"读取已有树文件: {tree_path}")
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    validation_issues = _validate_tree(payload)
    if validation_issues and progress is not None:
        for issue in validation_issues[:20]:
            progress(f"树验证警告: {issue}")
        if len(validation_issues) > 20:
            progress(f"树验证警告: ... 还有 {len(validation_issues) - 20} 个问题")

    (
        node_ids,
        chunk_ids,
        node_vectors,
        chunk_vectors,
    ) = _build_vectors_for_payload(
        payload,
        clients,
        progress=progress,
    )
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"写入向量文件: {vectors_path}")
    np.savez_compressed(
        vectors_path,
        node_ids=np.asarray(node_ids),
        chunk_ids=np.asarray(chunk_ids),
        node_vectors=node_vectors,
        chunk_vectors=chunk_vectors,
    )
    if progress is not None:
        progress("向量重建完成，树文件未改写")
    return tree_path, vectors_path


def build_and_save_index(
    settings: Settings,
    clients: OnlineClients,
    *,
    rule_options: BuildRuleOptions | None = None,
    tree_path: Path | None = None,
    vectors_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    options = rule_options or BuildRuleOptions()
    storage_dir = _rule_storage_dir(settings, options)
    storage_dir.mkdir(parents=True, exist_ok=True)
    tree_file = tree_path or (storage_dir / "regtree_tree.json")
    vectors_file = vectors_path or (storage_dir / "regtree_vectors.npz")
    tree_file.parent.mkdir(parents=True, exist_ok=True)
    vectors_file.parent.mkdir(parents=True, exist_ok=True)

    if progress is not None:
        progress("开始解析文档并构建树结构")
    payload = build_tree_payload(settings, rule_options=options, clients=clients, progress=progress)
    node_ids = list(payload["nodes"].keys())
    chunk_ids = list(payload["chunks"].keys())
    if progress is not None:
        progress(f"树结构解析完成: nodes={len(node_ids)}, chunks={len(chunk_ids)}")
    validation_issues = _validate_tree(payload)
    if validation_issues and progress is not None:
        for issue in validation_issues[:20]:
            progress(f"树验证警告: {issue}")
        if len(validation_issues) > 20:
            progress(f"树验证警告: ... 还有 {len(validation_issues) - 20} 个问题")
    (
        node_ids,
        chunk_ids,
        node_vectors,
        chunk_vectors,
    ) = _build_vectors_for_payload(
        payload,
        clients,
        progress=progress,
    )

    if progress is not None:
        progress(f"写入树文件: {tree_file}")
    tree_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if progress is not None:
        progress(f"写入向量文件: {vectors_file}")
    np.savez_compressed(
        vectors_file,
        node_ids=np.asarray(node_ids),
        chunk_ids=np.asarray(chunk_ids),
        node_vectors=node_vectors,
        chunk_vectors=chunk_vectors,
    )
    options = rule_options or BuildRuleOptions()
    checkpoint_path = options.checkpoint_path or _default_llm_checkpoint_file(settings, options)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        if progress is not None:
            progress(f"清理 LLM checkpoint: {checkpoint_path}")
    if progress is not None:
        progress("索引构建完成")
    return tree_file, vectors_file


def _progress(message: str) -> None:
    print(f"[tree_index] {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tree index directly from tree_index.py")
    parser.add_argument("--rule", help="Override extraction rule for all input files")
    parser.add_argument("--rule-map", help="JSON file mapping filename patterns to rule names")
    parser.add_argument("--rule-file", help="JSON file defining extraction rule profiles")
    parser.add_argument("--dataset-name", help="Dataset-specific rules directory name under rules/")
    parser.add_argument("--dataset-path", help="Dataset file path used to derive rules/<dataset_name>/")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted LLM tree build from checkpoint (enabled by default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and rebuild from scratch",
    )
    parser.add_argument(
        "--checkpoint-file",
        help="Optional checkpoint file path for interrupted LLM tree builds",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    settings = Settings.load(workspace_root)
    clients = OnlineClients(settings)
    tree_path, vectors_path = build_and_save_index(
        settings,
        clients,
        rule_options=BuildRuleOptions(
            override_rule=args.rule,
            rule_map_path=Path(args.rule_map).resolve() if args.rule_map else None,
            rule_file=Path(args.rule_file).resolve() if args.rule_file else None,
            dataset_name=args.dataset_name,
            dataset_path=Path(args.dataset_path).resolve() if args.dataset_path else None,
            resume=False if args.no_resume else True,
            checkpoint_path=Path(args.checkpoint_file).resolve() if args.checkpoint_file else None,
        ),
        progress=_progress,
    )
    print(
        json.dumps(
            {
                "tree_path": str(tree_path),
                "vectors_path": str(vectors_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
