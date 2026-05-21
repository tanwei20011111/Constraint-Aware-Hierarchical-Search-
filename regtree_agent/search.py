
from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Settings
from .online import OnlineClients
from .prompts import (
    ANCHOR_SYSTEM_PROMPT,
    ANCHOR_TASK_TEMPLATE,
    ANSWER_EXPAND_NOTE,
    ANSWER_SYSTEM_PROMPT_TEMPLATE,
    ANSWER_TASK_TEMPLATE,
    EXTRACT_FINAL_CODE_SYSTEM_PROMPT,
    EXTRACT_FINAL_CODE_TASK_TEMPLATE,
    PLAN_NEXT_ROUND_SYSTEM_PROMPT,
    PLAN_NEXT_ROUND_TASK,
    SELECT_FIELD_MODE_NOTES,
    SELECT_SYSTEM_PROMPT_TEMPLATE,
    SELECT_TASK_TEMPLATE,
)

CandidateFieldMode = str

CANDIDATE_FIELD_MODES = {
    "title_only",
    "title_evidence",
    "title_text",
    "full",
}


def _cosine_similarity_with_norms(
    query: np.ndarray,
    matrix: np.ndarray,
    *,
    query_norm: float,
    matrix_norm: np.ndarray,
) -> np.ndarray:
    denom = np.clip(query_norm * matrix_norm, 1e-8, None)
    return np.dot(matrix, query) / denom


def _truncate(text: str, max_chars: int = 1000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_node_label(node: dict[str, Any]) -> str:
    code = str(node.get("code", "")).strip()
    title = str(node.get("title", "")).strip()
    node_type = str(node.get("node_type", "")).strip()
    prefix = f"{code} " if code else ""
    suffix = f" [{node_type}]" if node_type else ""
    return f"{prefix}{title}{suffix}".strip()


def _format_candidate_brief(candidates: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for item in candidates[:limit]:
        label = _format_node_label(item)
        retrieval_score = float(item.get("retrieval_score", item["score"]))
        source = str(item.get("retrieval_source", "child"))
        parts.append(f"{label} (source={source}, retrieval={retrieval_score:.3f})")
    suffix = ""
    if len(candidates) > limit:
        suffix = f"; ... +{len(candidates) - limit} more"
    return "; ".join(parts) + suffix


def _candidate_package(candidate: dict[str, Any]) -> dict[str, Any]:
    # Gamma(u) is the structured candidate package described in the paper.
    # We keep the output field names close to the math symbols so the runtime
    # trace can be compared directly with the formula.
    return {
        "id_u": candidate["id"],
        "code_u": candidate["code"],
        "title_u": candidate["title"],
        "type_u": candidate["node_type"],
        "source_u": candidate.get("retrieval_source", "child"),
        "s_u": candidate["score"],
        "N_u": candidate["notes"],
        "X_u": candidate["exclusions"],
        "D_u": candidate["definitions"],
        "E_u_given_x": candidate["evidence"],
    }


def _normalize_candidate_field_mode(mode: str | None) -> CandidateFieldMode:
    normalized = (mode or "full").strip().lower().replace("-", "_")
    aliases = {
        "title": "title_only",
        "titleonly": "title_only",
        "title_only": "title_only",
        "title_evidence": "title_evidence",
        "title_evidence_excerpt": "title_evidence",
        "title_text": "title_text",
        "full_candidate": "full",
        "full": "full",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CANDIDATE_FIELD_MODES:
        allowed = ", ".join(sorted(CANDIDATE_FIELD_MODES))
        raise ValueError(f"未知 candidate_field_mode={mode!r}，可选值: {allowed}")
    return normalized


def _compact_evidence_excerpt(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.get("chunk_id", ""),
            "pages": item.get("pages", ""),
            "excerpt": item.get("excerpt", ""),
        }
        for item in evidence
    ]


def _full_evidence_text(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.get("chunk_id", ""),
            "pages": item.get("pages", ""),
            "full_text": item.get("full_text", ""),
        }
        for item in evidence
    ]


def _llm_candidate_for_mode(candidate: dict[str, Any], mode: CandidateFieldMode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": candidate["id"],
        "code": candidate["code"],
        "title": candidate["title"],
        "node_type": candidate["node_type"],
    }
    if mode == "title_evidence":
        payload["evidence"] = _compact_evidence_excerpt(candidate.get("evidence", []))
    elif mode == "title_text":
        payload["text"] = candidate.get("text", "")
    elif mode == "full":
        payload.update(
            {
                "score": candidate.get("score", 0.0),
                "exclusions": candidate.get("exclusions", []),
                "text": candidate.get("text", ""),
                "definitions": candidate.get("definitions", []),
                "notes": candidate.get("notes", []),
                "evidence": _full_evidence_text(candidate.get("evidence", [])),
            }
        )
    return payload


def _serialize_llm_payload(system_prompt: str, prompt: dict[str, Any]) -> str:
    return json.dumps(
        {
            "system_prompt": system_prompt,
            "user_prompt": prompt,
        },
        ensure_ascii=False,
        separators=(',', ':'),
    )


def _serialize_json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def _vector_preview(vector: np.ndarray, limit: int = 8) -> list[float]:
    return [round(float(value), 6) for value in vector[:limit].tolist()]


class _LRUDict(OrderedDict):
    def __init__(self, maxsize: int = 256):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


def _confidence_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_hs_code(value: Any) -> str:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    if len(digits) >= 6:
        return digits[:6]
    if len(digits) == 5:
        return digits.zfill(6)
    return digits.ljust(6, "0")


def _format_answer_text(value: Any, max_chars: int = 1200) -> str:
    return _truncate(str(value), max_chars=max_chars)

_TASK_SUFFIX_RE = re.compile(
    r"(的?(?:\d+位)?\s*hs码|的?(?:\d+位)?\s*HS码|的?\s*商品归类|的?\s*归类|的?\s*海关编码)\s*$",
    re.IGNORECASE,
)
_CHAPTER_DIGIT_RE = re.compile(r"第\s*(\d{1,2})\s*章")
_CHAPTER_CN_RE = re.compile(r"第\s*([零〇一二两三四五六七八九十百]+)\s*章")
_CODE_TOKEN_RE = re.compile(r"(?<!\d)\d{4,6}(?!\d)|(?:CN\s*)?\d{2}\.\d{2}(?:\.\d{2})?", re.IGNORECASE)
_CODE_PHRASE_RE = re.compile(
    r"(?:CN\s*)?\d{2}\.\d{2}(?:\.\d{2})?|(?:品目|子目|税号|编码|HS编码|HS码|CN)\s*\d{4,6}",
    re.IGNORECASE,
)
_ATTRIBUTE_TERM_RE = re.compile(
    r"(\d|工业级|食品级|医药级|试剂级|黏度|粘度|规格|型号|含量|浓度|纯度|包装|净重|容量|尺寸|长度|宽度|厚度|电压|功率|频率|温度|压力|颜色|外观|等级|归类|HS码|HS编码|编码|6位)",
    re.IGNORECASE,
)
_GENERIC_NON_ANCHOR_TERMS = {
    "化学改性",
    "改性",
    "阳离子改性",
    "工业级",
    "归类",
    "品目",
    "子目",
    "编码",
    "hs码",
    "hs编码",
    "6位",
}
_STOP_WORDS = {"的6位hs码", "6位hs码", "6位", "hs码", "编码", "查找", "寻找", "归类"}
_STRUCTURAL_NODE_TYPES = {"root", "document", "reference"}
_GLOBAL_RECALL_EXCLUDED_NODE_TYPES = {"root", "document"}
_CANDIDATE_EVIDENCE_EXCERPT_MAX_CHARS = 400
_CANDIDATE_EVIDENCE_MATCH_CONTEXT_CHARS = 30
_ANSWER_EVIDENCE_MAX_CHARS = 30000



_JUDGMENT_ANCHOR_RE = re.compile(
    r"(是否允许|归类规则|含量限制|改变用途|关键判定标准|归类争议|对归类的影响|区别|影响|争议|标准)",
    re.IGNORECASE,
)
_PROCESS_DESCRIPTOR_RE = re.compile(
    r"(改性|处理|加工|调制|配制|混合|复配|提取|精制|包覆|涂覆|制得|制成|获得|添加|加入|用于|适于|作为|化学制|天然|人工)",
    re.IGNORECASE,
)
_AUXILIARY_DESCRIPTOR_RE = re.compile(
    r"(处理剂|活性剂|添加剂|助剂|辅料|载体|溶剂|催化剂|稳定剂|乳化剂|分散剂)",
    re.IGNORECASE,
)
_ANCHOR_SPLIT_RE = re.compile(
    r"[；;，,、/]|(?:由|经|通过|采用|用于|适于|作为|制得|制成|获得|含有|包含|以及|及|与|或|并|且|和)"
)
_FOLLOWUP_SPLIT_RE = re.compile(r"[；;，,。！？!?、]")
_ASCII_UNIT_RE = re.compile(
    r"[A-Za-z]{1,8}(?:/[A-Za-z]{1,8})?",
    re.IGNORECASE,
)
_ALNUM_MODEL_LIKE_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._-]{4,}", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"[\d\-./]+")
_REJECTION_REASON_RE = re.compile(
    r"(不匹配|不符合|排除|被.*排除|错误分支|应停止|重新考虑|重新定位|无法匹配|不应继续|不属于|当前节点.*不匹配)",
    re.IGNORECASE,
)
_CHINESE_NUMERAL_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _anchor_core_text(text: str) -> str:
    value = str(text)
    value = _CODE_TOKEN_RE.sub(" ", value)
    value = _ATTRIBUTE_TERM_RE.sub(" ", value)
    value = _PROCESS_DESCRIPTOR_RE.sub(" ", value)
    value = _AUXILIARY_DESCRIPTOR_RE.sub(" ", value)
    value = re.sub(r"(品目|子目|税号|章|类)", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[\d.%/<>≤≥+\-–~()（）]+", " ", value)
    value = _ASCII_UNIT_RE.sub(" ", value)
    return re.sub(r"\s+", "", value)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def _parse_chinese_chapter_number(text: str) -> int | None:
    value = str(text).strip()
    if not value:
        return None
    if value == "十":
        return 10
    if "百" in value:
        head, tail = value.split("百", 1)
        head_value = _CHINESE_NUMERAL_MAP.get(head, 1 if not head else None)
        if head_value is None:
            return None
        tail_value = _parse_chinese_chapter_number(tail) or 0
        return head_value * 100 + tail_value
    if "十" in value:
        head, tail = value.split("十", 1)
        head_value = _CHINESE_NUMERAL_MAP.get(head, 1 if not head else None)
        if head_value is None:
            return None
        tail_value = _CHINESE_NUMERAL_MAP.get(tail, 0) if tail else 0
        return head_value * 10 + tail_value
    total = 0
    for ch in value:
        digit = _CHINESE_NUMERAL_MAP.get(ch)
        if digit is None:
            return None
        total = total * 10 + digit
    return total or None


def _extract_hs_chapter_code(node: dict[str, Any]) -> str:
    title = str(node.get("title", ""))
    match = _CHAPTER_DIGIT_RE.search(title)
    if match:
        return match.group(1).zfill(2)
    match = _CHAPTER_CN_RE.search(title)
    if match:
        number = _parse_chinese_chapter_number(match.group(1))
        if number is not None:
            return str(number).zfill(2)
    code = "".join(ch for ch in str(node.get("code", "")) if ch.isdigit())
    if len(code) >= 4:
        return code[:2]
    return ""


def _normalize_query_text(text: str) -> str:
    value = str(text).strip()
    value = _TASK_SUFFIX_RE.sub("", value)
    value = value.replace("；", " ").replace(";", " ")
    value = value.replace("，", " ").replace(",", " ")
    value = value.replace("：", " ").replace(":", " ")
    value = value.replace("（", " ").replace("）", " ")
    value = value.replace("(", " ").replace(")", " ")
    return re.sub(r"\s+", " ", value).strip()


def _compose_followup_query(original_query: str, planner_trace: dict[str, Any]) -> str:
    action = str(planner_trace.get("action", "")).strip() or "refine_query"
    base = _normalize_query_text(original_query)
    focus = _normalize_query_text(planner_trace.get("focus", ""))
    next_query = _normalize_query_text(planner_trace.get("next_query", ""))
    focus_terms = _normalize_text_list(planner_trace.get("focus_terms", []))
    original_fragments = _split_query_fragments(base, max_fragment_chars=32)
    display_base_terms = [fragment for fragment in original_fragments if len(fragment) >= 2][:4]

    def compact_key(text: str) -> str:
        return re.sub(r"\s+", "", _normalize_query_text(text)).casefold()

    def append_unique(
        output: list[str],
        seen_keys: set[str],
        fragments: list[str],
        *,
        max_items: int,
        max_chars: int,
    ) -> None:
        for fragment in fragments:
            cleaned = _normalize_query_text(fragment)
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars].strip()
            key = compact_key(cleaned)
            token_keys = [compact_key(token) for token in cleaned.split() if compact_key(token)]
            if token_keys and all(token_key in seen_keys for token_key in token_keys):
                continue
            if not cleaned or not key or key in seen_keys:
                continue
            seen_keys.add(key)
            output.append(cleaned)
            if len(output) >= max_items:
                return

    if action == "compare_branches":
        compare_fragments: list[str] = []
        seen_compare: set[str] = set()
        append_unique(compare_fragments, seen_compare, focus_terms, max_items=5, max_chars=40)
        if not compare_fragments:
            append_unique(compare_fragments, seen_compare, [focus], max_items=5, max_chars=48)
        if len(compare_fragments) < 5:
            append_unique(compare_fragments, seen_compare, display_base_terms[:2], max_items=5, max_chars=32)
        compare_query = " ".join(compare_fragments).strip()
        if compare_query:
            return compare_query[:220].strip()

    if action == "switch_sibling":
        sibling_fragments: list[str] = []
        seen_sibling: set[str] = set()
        append_unique(sibling_fragments, seen_sibling, focus_terms[:4], max_items=5, max_chars=36)
        if not sibling_fragments:
            append_unique(sibling_fragments, seen_sibling, [focus], max_items=5, max_chars=48)
        if len(sibling_fragments) < 5:
            append_unique(sibling_fragments, seen_sibling, display_base_terms[:2], max_items=5, max_chars=32)
        sibling_query = " ".join(sibling_fragments).strip()
        if sibling_query:
            return sibling_query[:220].strip()

    fragments: list[str] = []
    seen: set[str] = set()
    total_chars = 0

    for term in focus_terms[:5]:
        fragment = _normalize_query_text(term)
        if len(fragment) > 32:
            fragment = fragment[:32].strip()
        key = compact_key(fragment)
        if not fragment or not key or key in seen:
            continue
        seen.add(key)
        fragments.append(fragment)
        total_chars += len(fragment) + 1

    for source_text, max_fragments, max_fragment_chars in [
        (next_query, 4, 32),
        (focus, 4, 48),
        (base, 3, 32),
    ]:
        source_fragments = _split_query_fragments(source_text, max_fragment_chars=max_fragment_chars)
        kept = 0
        for fragment in source_fragments:
            fragment = _normalize_query_text(fragment)
            key = compact_key(fragment)
            token_keys = [compact_key(token) for token in fragment.split() if compact_key(token)]
            if token_keys and all(token_key in seen for token_key in token_keys):
                continue
            if not fragment or key in seen:
                continue
            if total_chars and total_chars + len(fragment) + 1 > 220:
                continue
            seen.add(key)
            fragments.append(fragment)
            total_chars += len(fragment) + 1
            kept += 1
            if kept >= max_fragments:
                break

    return " ".join(fragments).strip()


def _split_query_fragments(text: str, *, max_fragment_chars: int = 48) -> list[str]:
    normalized = _normalize_query_text(text)
    if not normalized:
        return []

    fragments: list[str] = []
    seen: set[str] = set()
    primary_parts = [
        _normalize_query_text(part)
        for part in _FOLLOWUP_SPLIT_RE.split(normalized)
        if _normalize_query_text(part)
    ]

    for part in primary_parts:
        secondary_parts = [part]
        if len(part) > max_fragment_chars:
            secondary_parts = [
                _normalize_query_text(item)
                for item in _ANCHOR_SPLIT_RE.split(part)
                if _normalize_query_text(item)
            ] or [part]
        for fragment in secondary_parts:
            if len(fragment) > max_fragment_chars:
                fragment = fragment[:max_fragment_chars].strip()
            if len(fragment) < 2:
                continue
            key = fragment.casefold()
            if key in seen:
                continue
            seen.add(key)
            fragments.append(fragment)
    return fragments


def _query_fact_fingerprint(text: str) -> set[str]:
    normalized = _normalize_query_text(text).casefold()
    if not normalized:
        return set()
    return {
        token
        for token in re.split(r"\s+", normalized)
        if len(token) >= 2 and token not in _STOP_WORDS
    }


def _candidate_anchor_terms(
    query: str,
    *,
    core_terms: list[str] | None = None,
    fallback_terms: list[str] | None = None,
) -> list[str]:
    def normalize_terms(candidates: list[str]) -> list[str]:
        normalized_candidates: list[str] = []
        seen: set[str] = set()
        for term in candidates:
            cleaned = re.sub(r"\s+", "", term)
            if not cleaned:
                continue
            if _CODE_TOKEN_RE.fullmatch(cleaned):
                continue
            if cleaned.casefold() in _GENERIC_NON_ANCHOR_TERMS:
                continue
            if _JUDGMENT_ANCHOR_RE.search(cleaned):
                continue
            anchor = _anchor_core_text(cleaned)
            if not anchor:
                if (
                    _ATTRIBUTE_TERM_RE.search(cleaned)
                    or _PROCESS_DESCRIPTOR_RE.search(cleaned)
                    or _AUXILIARY_DESCRIPTOR_RE.search(cleaned)
                ):
                    continue
                anchor = cleaned
            if len(anchor) < 2:
                continue
            if len(anchor) == 2 and len(cleaned) >= 6:
                continue
            key = anchor.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized_candidates.append(anchor)
        normalized_candidates.sort(key=len, reverse=True)
        return normalized_candidates[:6]

    primary_candidates = [str(item).strip() for item in (core_terms or []) if str(item).strip()]
    normalized_candidates = normalize_terms(primary_candidates)
    if normalized_candidates:
        return normalized_candidates

    fallback_candidates = [str(item).strip() for item in (fallback_terms or []) if str(item).strip()]
    normalized_candidates = normalize_terms(fallback_candidates)
    if normalized_candidates:
        return normalized_candidates

    query_candidates: list[str] = []
    normalized_query = _normalize_query_text(query)
    base_segments = [
        segment.strip()
        for segment in re.split(r"[；;，,、]", normalized_query)
        if segment.strip()
    ]
    for segment in base_segments:
        if len(segment) <= 16:
            query_candidates.append(segment)
        query_candidates.extend(
            piece.strip()
            for piece in _ANCHOR_SPLIT_RE.split(segment)
            if piece.strip()
        )

    return normalize_terms(query_candidates)


def _has_anchor_support(
    anchors: list[str],
    *,
    retrieved_nodes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> bool:
    if not anchors:
        return True
    support_texts = [
        re.sub(r"\s+", "", str(node.get("title", "")) + str(node.get("text", "")))
        for node in retrieved_nodes
    ]
    support_texts.extend(
        re.sub(
            r"\s+",
            "",
            str(item.get("title", ""))
            + str(item.get("excerpt", ""))
            + str(item.get("full_text", "")),
        )
        for item in evidence
    )
    return any(anchor and any(anchor in text for text in support_texts) for anchor in anchors)


def _enforce_answer_support(
    *,
    query: str,
    answer_payload: dict[str, Any],
    retrieved_nodes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    core_terms: list[str] | None = None,
    fallback_terms: list[str] | None = None,
) -> dict[str, Any]:
    anchors = _candidate_anchor_terms(
        query,
        core_terms=core_terms,
        fallback_terms=fallback_terms,
    )
    if _has_anchor_support(anchors, retrieved_nodes=retrieved_nodes, evidence=evidence):
        return answer_payload

    revised = dict(answer_payload)
    revised["confidence"] = min(_confidence_value(revised.get("confidence")), 0.25)
    return revised


def _normalize_code_list(values: Any, *, allowed_lengths: set[int]) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) not in allowed_lengths or digits in seen:
            continue
        seen.add(digits)
        normalized.append(digits)
    return normalized


def _normalize_text_list(values: Any, *, max_items: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_query_text(value)
        compact = re.sub(r"\s+", "", cleaned)
        if not compact or _NUMERIC_TOKEN_RE.fullmatch(compact):
            continue
        if _ALNUM_MODEL_LIKE_RE.fullmatch(compact) and not _contains_cjk(compact):
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= max_items:
            break
    return normalized


def _merge_term_lists(*term_lists: list[str], max_items: int = 12) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for term_list in term_lists:
        for term in term_list:
            cleaned = _normalize_query_text(term)
            compact = re.sub(r"\s+", "", cleaned).casefold()
            if not cleaned or not compact or compact in seen:
                continue
            seen.add(compact)
            merged.append(cleaned)
            if len(merged) >= max_items:
                return merged
    return merged


def _excerpt_around_terms(
    text: str,
    terms: list[str] | None,
    *,
    context_chars: int = _CANDIDATE_EVIDENCE_MATCH_CONTEXT_CHARS,
    fallback_chars: int = _CANDIDATE_EVIDENCE_EXCERPT_MAX_CHARS,
) -> str:
    source = str(text)
    for term in sorted((str(item).strip() for item in terms or []), key=len, reverse=True):
        if not term:
            continue
        index = source.find(term)
        if index < 0:
            continue
        start = max(0, index - context_chars)
        end = min(len(source), index + len(term) + context_chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(source) else ""
        return f"{prefix}{source[start:end].strip()}{suffix}"
    return _truncate(source, fallback_chars)


def _code_matches_avoid_targets(code: str, targets: list[str]) -> bool:
    return any(code == target or code.startswith(target) for target in targets)


@dataclass(slots=True)
class SearchArtifacts:
    tree_path: Path
    vectors_path: Path


@dataclass(slots=True)
class SearchPassResult:
    query: str
    retrieval_query: str
    final_node_id: str
    final_node: dict[str, Any]
    search_path: list[dict[str, Any]]
    retrieved_nodes: list[dict[str, Any]]
    alternatives: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer_payload: dict[str, Any]


class RegTreeSearcher:
    def __init__(self, settings: Settings, clients: OnlineClients, artifacts: SearchArtifacts):
        self.settings = settings
        self.clients = clients
        self.payload = json.loads(artifacts.tree_path.read_text(encoding="utf-8"))
        self.nodes: dict[str, dict[str, Any]] = self.payload["nodes"]
        self.chunks: dict[str, dict[str, Any]] = self.payload["chunks"]
        packed = np.load(artifacts.vectors_path, allow_pickle=True)
        self.node_ids = [str(item) for item in packed["node_ids"].tolist()]
        self.chunk_ids = [str(item) for item in packed["chunk_ids"].tolist()]
        self.node_vectors = packed["node_vectors"].astype(np.float32)
        self.chunk_vectors = packed["chunk_vectors"].astype(np.float32)
        self.node_vector_norms = np.linalg.norm(self.node_vectors, axis=1)
        self.chunk_vector_norms = np.linalg.norm(self.chunk_vectors, axis=1)
        self.node_id_to_index = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.chunk_id_to_index = {chunk_id: idx for idx, chunk_id in enumerate(self.chunk_ids)}
        self.query_embedding_cache: _LRUDict = _LRUDict(maxsize=256)
        self._pass_evidence_cache: dict[str, list[dict[str, Any]]] = {}
        self.parent_by_id: dict[str, str] = {}
        self.code_to_node_ids: dict[str, set[str]] = {}
        self.chapter_part_nodes_by_hs_code: dict[str, set[str]] = {}
        self.node_code_digits: dict[str, str] = {}
        for node_id, node in self.payload["nodes"].items():
            code = "".join(ch for ch in str(node.get("code", "")) if ch.isdigit())
            self.node_code_digits[node_id] = code
            if code:
                self.code_to_node_ids.setdefault(code, set()).add(node_id)
            if str(node.get("node_type", "")) == "chapter":
                hs_chapter = _extract_hs_chapter_code(node)
                if hs_chapter:
                    self.chapter_part_nodes_by_hs_code.setdefault(hs_chapter, set()).add(node_id)
            for child_id in node.get("children", []):
                self.parent_by_id[str(child_id)] = str(node_id)
        self.global_recall_node_ids = [
            node_id
            for node_id in self.node_ids
            if str(self.nodes.get(node_id, {}).get("node_type", "")) not in _GLOBAL_RECALL_EXCLUDED_NODE_TYPES
            and node_id in self.node_id_to_index
        ]
        self.global_recall_indices = [self.node_id_to_index[node_id] for node_id in self.global_recall_node_ids]

    def _normalize_node_id_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            node_id = str(value).strip()
            if not node_id or node_id not in self.nodes or node_id in seen:
                continue
            seen.add(node_id)
            normalized.append(node_id)
        return normalized

    def _collect_subtree_ids(self, node_ids: list[str] | set[str]) -> set[str]:
        collected: set[str] = set()
        stack = [node_id for node_id in node_ids if node_id]
        while stack:
            current = str(stack.pop())
            if current in collected or current not in self.nodes:
                continue
            collected.add(current)
            stack.extend(str(child_id) for child_id in self.nodes[current].get("children", []))
        return collected

    def _ancestor_chain(self, node_id: str) -> list[str]:
        root_id = str(self.payload["root_id"])
        chain: list[str] = []
        current = str(node_id)
        while current != root_id and current in self.parent_by_id:
            current = str(self.parent_by_id[current])
            chain.append(current)
        chain.reverse()
        return chain

    def _resolve_avoid_hit_ids(
        self,
        avoid_codes: list[str] | None,
        avoid_node_ids: list[str] | None = None,
    ) -> set[str]:
        hit_ids: set[str] = set()
        for code in avoid_codes or []:
            hit_ids.update(self._collect_subtree_ids(self.code_to_node_ids.get(code, set())))
        if avoid_node_ids:
            hit_ids.update(avoid_node_ids)
        return hit_ids

    def _candidate_child_ids(self, parent_id: str) -> list[str]:
        child_ids = [str(child_id) for child_id in self.nodes[parent_id]["children"]]
        parent_node = self.nodes[parent_id]
        if str(parent_node.get("node_type", "")) in _STRUCTURAL_NODE_TYPES:
            return self._expand_structural_child_ids(child_ids)
        if str(parent_node.get("node_type", "")) != "chapter":
            return child_ids
        hs_chapter = _extract_hs_chapter_code(parent_node)
        if not hs_chapter:
            return child_ids
        seen = {str(child_id) for child_id in child_ids}
        for sibling_id in self.chapter_part_nodes_by_hs_code.get(hs_chapter, set()):
            if sibling_id == parent_id:
                continue
            for child_id in self.nodes[sibling_id].get("children", []):
                child_key = str(child_id)
                if child_key in seen:
                    continue
                seen.add(child_key)
                child_ids.append(child_key)
        return child_ids

    def _expand_structural_child_ids(self, child_ids: list[str]) -> list[str]:
        expanded: list[str] = []
        seen_nodes: set[str] = set()
        seen_output: set[str] = set()
        stack = list(reversed(child_ids))
        while stack:
            child_id = str(stack.pop())
            if child_id in seen_nodes or child_id not in self.nodes:
                continue
            seen_nodes.add(child_id)
            child_node = self.nodes[child_id]
            if str(child_node.get("node_type", "")) in _STRUCTURAL_NODE_TYPES:
                stack.extend(reversed([str(item) for item in child_node.get("children", [])]))
                continue
            if child_id in seen_output:
                continue
            seen_output.add(child_id)
            expanded.append(child_id)
        return expanded

    def _top_global_candidates(
        self,
        *,
        query_vector: np.ndarray,
        query_norm: float,
        exclude_ids: set[str],
        avoid_codes: list[str] | None = None,
        avoid_hit_ids: set[str] | None = None,
        top_k: int = 0,
        precomputed_global_sims: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.global_recall_indices:
            return []

        if precomputed_global_sims is not None:
            node_sims = precomputed_global_sims
        else:
            node_sims = _cosine_similarity_with_norms(
                query_vector,
                self.node_vectors[self.global_recall_indices],
                query_norm=query_norm,
                matrix_norm=self.node_vector_norms[self.global_recall_indices],
            )
        node_sims = np.asarray(node_sims, dtype=np.float32)
        ranked_indices = np.argsort(node_sims)[::-1]
        scan_limit = min(len(ranked_indices), max(top_k * 12, top_k + 20))
        selected: list[dict[str, Any]] = []
        avoid_hit_ids = avoid_hit_ids or set()
        avoid_codes_list = avoid_codes or []
        for rank in range(scan_limit):
            local_index = int(ranked_indices[rank])
            node_id = self.global_recall_node_ids[local_index]
            if node_id in exclude_ids or node_id in avoid_hit_ids:
                continue
            code = self.node_code_digits.get(node_id, "")
            if avoid_codes_list and _code_matches_avoid_targets(code, avoid_codes_list):
                continue
            selected.append(
                {
                    "id": node_id,
                    "semantic_score": float(node_sims[local_index]),
                    "node_score": float(node_sims[local_index]),
                    "retrieval_source": "global",
                }
            )
            if len(selected) >= top_k:
                break
        return selected

    def _embed_query(self, query: str, progress: Callable[[str], None] | None = None) -> np.ndarray:
        cache_key = _normalize_query_text(query) or query.strip()
        cached = self.query_embedding_cache.get(cache_key)
        if cached is not None:
            if progress is not None:
                progress("query embedding: cache hit")
            return cached
        vector = self.clients.embed_texts([cache_key], progress=progress, label="query embedding")[0]
        cached = np.asarray(vector, dtype=np.float32)
        self.query_embedding_cache[cache_key] = cached
        return cached

    def _llm_extract_anchor_terms_with_trace(
        self,
        query: str,
        *,
        progress: Callable[[str], None] | None = None,
        print_llm_inputs: bool = False,
    ) -> dict[str, Any]:
        system_prompt = ANCHOR_SYSTEM_PROMPT
        prompt = {
            "task": ANCHOR_TASK_TEMPLATE,
            "query": query,
            "examples": [
                {
                    "query": "阳离子改性瓜尔胶；由瓜尔豆通过化学改性处理获得；黏度300-1000 mPa.s；工业级",
                    "retrieval_query": "阳离子改性瓜尔胶 瓜尔豆 化学改性 工业级 黏度",
                    "anchor_terms": ["阳离子改性瓜尔胶", "瓜尔豆", "化学改性"],
                    "constraint_terms": ["黏度300-1000 mPa.s", "工业级"],
                }
            ],
            "output_schema": {
                "retrieval_query": "一行检索短语，短而具体，保留商品本体和关键限定条件",
                "anchor_terms": "字符串数组，1到8个短语；商品本体、来源材料、制得来源、关键工艺、直接类目词",
                "constraint_terms": "字符串数组，0到8个短语；规格、用途、材质、形态、等级、含量、型号、其他等限定条件",
                "reason": "一句中文说明为什么保留这些检索词",
            },
        }
        if print_llm_inputs and progress is not None:
            progress("llm[anchor_terms] request -> " + _serialize_llm_payload(system_prompt, prompt))
        payload = self.clients.chat_json(system_prompt, json.dumps(prompt, ensure_ascii=False, separators=(',', ':')))
        if print_llm_inputs and progress is not None:
            progress("llm[anchor_terms] response -> " + _serialize_json_payload(payload))
        return payload

    def _node_brief(self, node_id: str, text_limit: int = 800) -> dict[str, Any]:
        node = self.nodes[node_id]
        return {
            "id": node_id,
            "code": node["code"],
            "title": node["title"],
            "node_type": node["node_type"],
            "text": _truncate(node["text"], text_limit),
        }

    def _top_child_candidates(
        self,
        parent_id: str,
        query_text: str,
        query_vector: np.ndarray,
        query_norm: float,
        anchor_terms: list[str] | None = None,
        avoid_codes: list[str] | None = None,
        avoid_hit_ids: set[str] | None = None,
        top_k: int = 5,
        global_top_k: int = 0,
        visited_node_ids: list[str] | None = None,
        precomputed_global_sims: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """获取当前节点的候选子节点。

        候选来源有二：
          1. child  —— 当前节点的直接子节点（RegTree 结构展开）
          2. global —— 全局语义检索（跨整棵树）

        当 global_top_k > 0 时会启用全局检索；传入 0 则跳过。
        在新的搜索策略中，只有第一层（depth=0）会传入 global_top_k > 0，
        后续层传入 global_top_k=0，即只沿 RegTree 树结构向下展开子节点。
        不使用关键词召回。
        """
        child_ids = self._candidate_child_ids(parent_id)
        if avoid_hit_ids or avoid_codes:
            child_ids = [
                child_id
                for child_id in child_ids
                if child_id not in (avoid_hit_ids or set())
                and not _code_matches_avoid_targets(
                    self.node_code_digits.get(child_id, ""),
                    avoid_codes or [],
                )
            ]
        light_scored: list[dict[str, Any]] = []
        child_selection_limit = min(len(child_ids), top_k) if child_ids else 0
        if child_ids:
            indices = [self.node_id_to_index[child_id] for child_id in child_ids]
            node_sims = _cosine_similarity_with_norms(
                query_vector,
                self.node_vectors[indices],
                query_norm=query_norm,
                matrix_norm=self.node_vector_norms[indices],
            )
            candidate_pool_limit = min(len(child_ids), max(20, child_selection_limit * 4, top_k * 4))
            child_scored: list[dict[str, Any]] = []
            for i, child_id in enumerate(child_ids):
                node_score = float(node_sims[i])
                child_scored.append(
                    {
                        "id": child_id,
                        "semantic_score": node_score,
                        "node_score": node_score,
                        "retrieval_source": "child",
                    }
                )
            child_scored.sort(key=lambda item: item["semantic_score"], reverse=True)
            light_scored.extend(child_scored[:candidate_pool_limit])

        global_scored = self._top_global_candidates(
            query_vector=query_vector,
            query_norm=query_norm,
            exclude_ids={parent_id, *child_ids, *(visited_node_ids or [])},
            avoid_codes=avoid_codes,
            avoid_hit_ids=avoid_hit_ids,
            top_k=global_top_k,
            precomputed_global_sims=precomputed_global_sims,
        )
        light_scored.extend(global_scored)
        if not light_scored:
            return []

        light_by_id = {str(item["id"]): item for item in light_scored}
        scored = []
        for node_id, light_item in light_by_id.items():
            node = self.nodes[node_id]
            evidence = self._candidate_evidence(
                node_id,
                query_vector,
                query_norm=query_norm,
                limit=1,
                excerpt_terms=anchor_terms,
            )
            semantic_score = float(light_item["semantic_score"])
            node_text = _truncate(node["text"], 900)
            scored.append(
                {
                    "id": node_id,
                    "code": node["code"],
                    "title": node["title"],
                    "node_type": node["node_type"],
                    "score": semantic_score,
                    "retrieval_score": semantic_score,
                    "semantic_score": float(semantic_score),
                    "node_score": light_item.get("node_score"),
                    "retrieval_source": light_item.get("retrieval_source", "child"),
                    "notes": node["notes"][:3],
                    "exclusions": node["exclusions"][:3],
                    "definitions": node["definitions"][:3],
                    "text": node_text,
                    "evidence": evidence,
                }
            )
        if not scored:
            return []

        semantic_ranked = sorted(scored, key=lambda item: item["semantic_score"], reverse=True)

        selection_limit = min(
            len(scored),
            max(child_selection_limit, top_k) + max(global_top_k, 0),
        )
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in semantic_ranked:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            selected.append(item)
            if len(selected) >= selection_limit:
                break
        return selected

    def _emit_similarity_trace(
        self,
        *,
        parent_id: str,
        retrieval_query: str,
        query_vector: np.ndarray,
        candidates: list[dict[str, Any]],
        depth: int,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if progress is None or not candidates:
            return
        parent_node = self.nodes[parent_id]
        payload = {
            "formula": "s(u|v_t^(r),x)=cos(e(q_ret^(r)),e(u)), u in Child(v_t^(r))",
            "depth": depth,
            "q_ret^(r)": retrieval_query,
            "e(q_ret^(r))[:8]": _vector_preview(query_vector),
            "v_t^(r)": {
                "id": parent_id,
                "code": parent_node["code"],
                "title": parent_node["title"],
                "node_type": parent_node["node_type"],
            },
            "Child(v_t^(r))": [
                {
                    "u": {
                        "id": candidate["id"],
                        "code": candidate["code"],
                        "title": candidate["title"],
                        "node_type": candidate["node_type"],
                        "retrieval_source": candidate.get("retrieval_source", "child"),
                    },
                    "e(u)[:8]": _vector_preview(
                        self.node_vectors[self.node_id_to_index[candidate["id"]]]
                    ),
                    "s(u|v_t^(r),x)": round(float(candidate["score"]), 6),
                    "node_score": (
                        round(float(candidate["node_score"]), 6)
                        if candidate.get("node_score") is not None
                        else None
                    ),
                }
                for candidate in candidates
            ],
        }
        progress(f"depth={depth}: similarity trace -> {_serialize_json_payload(payload)}")

    def _candidate_evidence(
        self,
        node_id: str,
        query_vector: np.ndarray,
        query_norm: float,
        limit: int = 2,
        excerpt_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        cache_key = (node_id, tuple(excerpt_terms or []))
        cached = self._pass_evidence_cache.get(cache_key)
        if cached is not None:
            return cached[:limit]
        node = self.nodes[node_id]
        evidence_chunk_ids = node["evidence_chunk_ids"]
        if not evidence_chunk_ids:
            return []
        valid_ids = []
        valid_indices = []
        for chunk_id in evidence_chunk_ids:
            idx = self.chunk_id_to_index.get(chunk_id)
            if idx is None:
                continue
            valid_ids.append(chunk_id)
            valid_indices.append(idx)
        if not valid_ids:
            return []
        scores = _cosine_similarity_with_norms(
            query_vector,
            self.chunk_vectors[valid_indices],
            query_norm=query_norm,
            matrix_norm=self.chunk_vector_norms[valid_indices],
        )
        scored: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(valid_ids):
            chunk = self.chunks[chunk_id]
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "title": chunk["title"],
                    "pages": f"{chunk['start_page']}-{chunk['end_page']}",
                    "score": float(scores[i]),
                    "excerpt": _excerpt_around_terms(chunk["text"], excerpt_terms),
                    "full_text": _truncate(str(chunk["text"]), max_chars=2000),
                }
            )
        if not scored:
            return []

        scored.sort(key=lambda item: item["score"], reverse=True)
        result = scored[:limit]
        self._pass_evidence_cache[cache_key] = result
        return result

    def _llm_select(
        self,
        *,
        query: str,
        parent_node: dict[str, Any],
        candidates: list[dict[str, Any]],
        candidate_field_mode: CandidateFieldMode = "full",
        avoid_codes: list[str] | None = None,
        avoid_node_ids: list[str] | None = None,
        avoid_hit_ids: set[str] | None = None,
        depth: int,
        progress: Callable[[str], None] | None = None,
        print_llm_inputs: bool = False,
        print_answer_trace: bool = False,
    ) -> dict[str, Any]:
        candidate_field_mode = _normalize_candidate_field_mode(candidate_field_mode)
        field_mode_note = SELECT_FIELD_MODE_NOTES[candidate_field_mode]
        system_prompt = SELECT_SYSTEM_PROMPT_TEMPLATE.format(field_mode_note=field_mode_note)
        llm_candidates = []
        for item in candidates:
            llm_candidates.append(_llm_candidate_for_mode(item, candidate_field_mode))
        prompt = {
            "task": SELECT_TASK_TEMPLATE.format(field_mode_note=field_mode_note),
            "query": query,
            "current_node": {
                "code": parent_node["code"],
                "title": parent_node["title"],
                "node_type": parent_node["node_type"],
            },
            "selection_context": {
                "note": "不要根据候选排列顺序做决定。",
            },
            "candidates": llm_candidates,
            "output_schema": {
                "candidate_scores": "对所有候选包中的 candidates 进行打分，评分依据是候选内容与 query 之间的匹配程度，且必须确保覆盖所有 candidates。",
                "selected_id": "必须是 candidates 中某个 id；如果 stop=true 也保留最优候选 id",
                "stop": "布尔值，是否在当前所选节点停止继续下钻",
                "confidence": "0 到 1 的浮点数；表示对 selected_id 这个选择动作的整体置信度。",
                "reason": "理由说明",
            },
        }
        if print_answer_trace and progress is not None:
            progress(
                f"select[depth={depth + 1}] formula input -> "
                + _serialize_json_payload(
                    {
                        "formula": "(v_{t+1}, sigma_t, rho_t, eta_t) = F_theta(x, v_t, {Gamma(u)}_{u in C_t(x)})",
                        "x": query,
                        "v_t": {
                            "code": parent_node["code"],
                            "title": parent_node["title"],
                            "node_type": parent_node["node_type"],
                        },
                        "C_t(x)": [
                            {
                                "u": {
                                    "id": item["id"],
                                    "code": item["code"],
                                    "title": item["title"],
                                    "node_type": item["node_type"],
                                },
                                "Gamma(u)": _candidate_package(item),
                            }
                            for item in candidates
                        ],
                    }
                )
            )
        if print_llm_inputs and progress is not None:
            progress(f"llm[select][depth={depth + 1}] request -> " + _serialize_llm_payload(system_prompt, prompt))
        response_payload = self.clients.chat_json(system_prompt, json.dumps(prompt, ensure_ascii=False, separators=(',', ':')))
        if print_llm_inputs and progress is not None:
            progress(f"llm[select][depth={depth + 1}] response -> " + _serialize_json_payload(response_payload))
        if print_answer_trace and progress is not None:
            progress(
                f"select[depth={depth + 1}] formula output -> "
                + _serialize_json_payload(
                    {
                        "formula": "(v_{t+1}, sigma_t, rho_t, eta_t) = F_theta(x, v_t, {Gamma(u)}_{u in C_t(x)})",
                        "v_{t+1}": response_payload.get("selected_id", ""),
                        "sigma_t": response_payload.get("stop", False),
                        "rho_t": _confidence_value(response_payload.get("confidence")),
                        "eta_t": response_payload.get("reason", ""),
                    }
                )
            )
        return response_payload

    def _emit_candidate_packages(
        self,
        *,
        parent_id: str,
        candidates: list[dict[str, Any]],
        depth: int,
        round_query: str,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if progress is None or not candidates:
            return
        parent_node = self.nodes[parent_id]
        payload = {
            "depth": depth,
            "query": round_query,
            "parent_node": {
                "id": parent_id,
                "code": parent_node["code"],
                "title": parent_node["title"],
                "node_type": parent_node["node_type"],
            },
            "candidate_packages": [_candidate_package(candidate) for candidate in candidates],
        }
        progress(f"depth={depth}: candidate packages -> {json.dumps(payload, ensure_ascii=False)}")

    def _coerce_selected_id(self, choice: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
        selected_id = str(choice.get("selected_id", "")).strip()
        if any(item["id"] == selected_id for item in candidates):
            return selected_id
        return str(candidates[0]["id"])

    def _coerce_candidate_match_scores(
        self,
        choice: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, float]:
        candidate_ids = {str(item["id"]) for item in candidates}
        scores: dict[str, float] = {}
        raw_scores = choice.get("candidate_scores", [])
        if isinstance(raw_scores, dict):
            iterable = [{"id": key, "match_score": value} for key, value in raw_scores.items()]
        elif isinstance(raw_scores, list):
            iterable = raw_scores
        else:
            iterable = []
        for item in iterable:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("id", "")).strip()
            if candidate_id not in candidate_ids:
                continue
            scores[candidate_id] = max(0.0, min(_confidence_value(item.get("match_score")), 1.0))

        selected_id = str(choice.get("selected_id", "")).strip()
        if not scores and selected_id in candidate_ids:
            # Backward compatibility for malformed/legacy model payloads. In the
            # normal path, match_score comes only from candidate_scores and is
            # intentionally not boosted by selection confidence.
            scores[selected_id] = self._coerce_selection_confidence(choice)
        for item in candidates:
            scores.setdefault(str(item["id"]), 0.0)
        return scores

    def _coerce_selection_confidence(self, choice: dict[str, Any]) -> float:
        return max(0.0, min(_confidence_value(choice.get("confidence")), 1.0))

    def _node_code_digits(self, node: dict[str, Any]) -> str:
        return "".join(ch for ch in str(node.get("code", "")) if ch.isdigit())

    def _generic_other_penalty(self, node: dict[str, Any]) -> int:
        title = re.sub(r"\s+", "", str(node.get("title", "")))
        return int(title in {"其他", "其他的"})

    def _representative_pass_code(self, item: SearchPassResult) -> str:
        for candidate in [item.final_node, *reversed(item.search_path), *item.retrieved_nodes]:
            code = "".join(ch for ch in str(candidate.get("code", "")) if ch.isdigit())
            if len(code) >= 4:
                return code[:4]
        return ""

    def _detect_conflicting_paths(self, passes: list[SearchPassResult]) -> dict[str, Any]:
        support_by_prefix: dict[str, float] = {}
        rounds_by_prefix: dict[str, list[int]] = {}
        for index, item in enumerate(passes, start=1):
            prefix = self._representative_pass_code(item)
            if not prefix:
                continue
            support = self._pass_aggregation_weight(item)
            support_by_prefix[prefix] = support_by_prefix.get(prefix, 0.0) + support
            rounds_by_prefix.setdefault(prefix, []).append(index)
        ranked = sorted(support_by_prefix.items(), key=lambda pair: pair[1], reverse=True)
        if len(ranked) < 2:
            return {
                "has_conflict": False,
                "competing_prefixes": [],
                "conflict_strength": 0.0,
                "rounds": {},
            }
        top_prefix, top_support = ranked[0]
        rival_prefixes = [
            (prefix, support)
            for prefix, support in ranked[1:]
            if prefix != top_prefix and support >= max(0.9, top_support * 0.45)
        ]
        if not rival_prefixes:
            return {
                "has_conflict": False,
                "competing_prefixes": [top_prefix],
                "conflict_strength": 0.0,
                "rounds": {top_prefix: rounds_by_prefix.get(top_prefix, [])},
            }
        competing = [top_prefix] + [prefix for prefix, _ in rival_prefixes]
        conflict_strength = sum(support for _, support in rival_prefixes) / max(top_support, 1e-6)
        return {
            "has_conflict": True,
            "competing_prefixes": competing,
            "conflict_strength": round(float(conflict_strength), 4),
            "rounds": {prefix: rounds_by_prefix.get(prefix, []) for prefix in competing},
        }

    def _rule_based_fallback_plan(
        self,
        *,
        original_query: str,
        current_query: str,
        result: SearchPassResult,
        new_node_ids: list[str],
        new_chunk_ids: list[str],
        avoid_node_ids: list[str],
        remaining_rounds: int,
    ) -> dict[str, Any] | None:
        if remaining_rounds <= 0:
            return None
        no_progress = not (new_node_ids or new_chunk_ids)
        low_confidence = _confidence_value(result.answer_payload.get("confidence")) < 0.85
        if not low_confidence:
            return None
        if not (no_progress and result.final_node_id in set(avoid_node_ids)):
            return None

        merged_avoid_nodes = list(dict.fromkeys([*(avoid_node_ids or []), result.final_node_id]))
        focus = "规则触发：当前避让分支无新增证据，尝试同层其他候选。"
        next_query = " ".join(
            fragment
            for fragment in [
                _normalize_query_text(original_query),
            ]
            if fragment
        ).strip()
        if not next_query:
            next_query = current_query
        return {
            "action": "switch_sibling",
            "continue_search": True,
            "next_query": next_query,
            "reason": "规则触发：当前避让分支无新增证据，改为换 sibling 分支重试。",
            "focus": focus,
            "focus_terms": [],
            "avoid_node_ids": merged_avoid_nodes,
        }

    def _collect_node_evidence(
        self,
        node_ids: list[str],
        query_vector: np.ndarray,
        query_norm: float,
        per_node_limit: int = 2,
        total_limit: int = 6,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for node_id in node_ids:
            for item in self._candidate_evidence(
                node_id,
                query_vector,
                query_norm=query_norm,
                limit=per_node_limit,
            ):
                existing = merged.get(item["chunk_id"])
                if existing is None or item["score"] > existing["score"]:
                    merged[item["chunk_id"]] = item
        ranked = sorted(
            merged.values(),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        return ranked[:total_limit]

    def _expand_evidence_for_answer(
        self,
        evidence: list[dict[str, Any]],
        max_chars: int = _ANSWER_EVIDENCE_MAX_CHARS,
    ) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for item in evidence:
            expanded_item = dict(item)
            chunk_id = str(item.get("chunk_id", "")).strip()
            chunk = self.chunks.get(chunk_id)
            if chunk is not None:
                full_text = str(chunk.get("text", ""))
                expanded_item["full_text"] = _truncate(full_text, max_chars=max_chars)
                expanded_item["full_text_chars"] = min(len(full_text), max_chars)
                expanded_item["full_text_truncated"] = len(full_text) > max_chars
            expanded.append(expanded_item)
        return expanded

    def _llm_plan_next_round(
        self,
        *,
        original_query: str,
        explored_rounds: list[dict[str, Any]],
        remaining_rounds: int,
        progress: Callable[[str], None] | None = None,
        print_llm_inputs: bool = False,
        print_answer_trace: bool = False,
    ) -> dict[str, Any]:
        system_prompt = PLAN_NEXT_ROUND_SYSTEM_PROMPT
        prompt = {
            "task": PLAN_NEXT_ROUND_TASK,
            "original_query": original_query,
            "remaining_rounds": remaining_rounds,
            "explored_rounds": explored_rounds,
            "output_schema": {
                "action": "枚举：refine_query、switch_sibling、compare_branches、stop",
                "continue_search": "布尔值，是否继续",
                "next_query": "若继续，给出下一轮更聚焦或改写后的查询；不要完整复述已有轮次 query；若停止可为空字符串",
                "reason": "一句中文理由，说明为什么继续或停止",
                "focus": "5字以内的关键方向词；若停止则为空字符串",
                "focus_terms": "若继续，给出3到6个短语数组，表示下一轮必须保留的判别词或区分标准词；无则空数组",
                "avoid_node_ids": "若继续，建议避免重复探索的节点id数组；仅可引用 explored_rounds 中出现过的 final_node_id 或 retrieved_node_ids；无则空数组。不要输出 avoid_codes。",
            },
        }
        if print_llm_inputs and progress is not None:
            progress("llm[plan_next_round] request -> " + _serialize_llm_payload(system_prompt, prompt))
        if print_answer_trace and progress is not None:
            progress(
                "planner formula input -> "
                + _serialize_json_payload(
                    {
                        "formula": "H^(r)={(q^(i),p^(i)(x),a^(i),E^(i)(x))}_{i=1}^r",
                        "r": len(explored_rounds),
                        "H^(r)": [
                            {
                                "i": item.get("round"),
                                "q^(i)": item.get("query", ""),
                                "p^(i)(x)": item.get("search_path", []),
                                "a^(i)": item.get("answer_summary", ""),
                                "E^(i)(x)": {
                                    "new_node_count": item.get("new_node_count", 0),
                                    "new_chunk_count": item.get("new_chunk_count", 0),
                                },
                            }
                            for item in explored_rounds
                        ],
                    }
                )
            )
        return self.clients.chat_json(system_prompt, json.dumps(prompt, ensure_ascii=False, separators=(',', ':')))

    def _llm_answer(
        self,
        *,
        query: str,
        retrieved_nodes: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        explored_rounds: list[dict[str, Any]] | None = None,
        progress: Callable[[str], None] | None = None,
        print_llm_inputs: bool = False,
        print_answer_trace: bool = False,
        answer_stage: str = "round",
        expand_from_root: bool = False,
    ) -> dict[str, Any]:
        expand_note = ANSWER_EXPAND_NOTE if expand_from_root else ""
        system_prompt = ANSWER_SYSTEM_PROMPT_TEMPLATE.format(expand_note=expand_note)
        prompt = {
            "task": ANSWER_TASK_TEMPLATE,
            "query": query,
            "retrieved_nodes": retrieved_nodes,
            "evidence": evidence,
            "explored_rounds": explored_rounds or [],
            "output_schema": {
                "reasoning": (
                    "中文推理过程：先汇总证据，再分析匹配/排除理由，最后得出编码结论。"
                ),
                "answer": "中文回答，必须受证据约束；证据充分时包含最终编码且最后一行尽量只输出最终编码，证据不足时不得输出编码",
                "final_code": "字符串。只能是空字符串、四位品目编码或六位子目编码。",
                "confidence": "0 到 1 的浮点数",
                "used_chunk_ids": "实际用到的 chunk_id 列表",
            },
        }
        if print_llm_inputs and progress is not None:
            progress("llm[answer] request -> " + _serialize_llm_payload(system_prompt, prompt))
        if print_answer_trace and progress is not None:
            progress(
                f"answer[{answer_stage}] formula input -> "
                + _serialize_json_payload(
                    {
                        "formula": "a^(r)=G_theta(x,p^(r)(x),E^(r)(x))",
                        "x": query,
                        "p^(r)(x)": retrieved_nodes,
                        "E^(r)(x)": evidence,
                    }
                )
            )
        response_payload = self.clients.chat_json(
            system_prompt,
            json.dumps(prompt, ensure_ascii=False, separators=(',', ':')),
        )
        if print_answer_trace and progress is not None:
            progress(
                f"answer[{answer_stage}] formula output -> "
                + _serialize_json_payload(
                    {
                        "formula": "a^(r)=G_theta(x,p^(r)(x),E^(r)(x))",
                        "a^(r)": {
                            "answer": _format_answer_text(response_payload.get("answer", "")),
                            "confidence": _confidence_value(response_payload.get("confidence")),
                            "used_chunk_ids": response_payload.get("used_chunk_ids", []),
                        },
                    }
                )
            )
        return response_payload

    def _llm_extract_final_hs_code(
        self,
        *,
        query: str,
        answer: str,
        retrieved_nodes: list[dict[str, Any]],
        explored_rounds: list[dict[str, Any]] | None = None,
        progress: Callable[[str], None] | None = None,
        print_llm_inputs: bool = False,
    ) -> str:
        system_prompt = EXTRACT_FINAL_CODE_SYSTEM_PROMPT
        task_text = EXTRACT_FINAL_CODE_TASK_TEMPLATE
        prompt = {
            "task": task_text,
            "query": query,
            "final_answer": answer,
            "candidate_nodes": [
                {
                    "code": node.get("code", ""),
                    "title": node.get("title", ""),
                    "node_type": node.get("node_type", ""),
                }
                for node in retrieved_nodes[:12]
            ],
            "explored_rounds": [
                {
                    "round": item.get("round"),
                    "final_node": item.get("final_node", {}),
                    "answer_summary": _truncate(str(item.get("answer_summary", "")), 200),
                }
                for item in (explored_rounds or [])[:8]
            ],
            "output_schema": {
                scheme.final_code_label: scheme.final_code_description,
                "reason": "一句中文说明依据。",
            },
        }
        if print_llm_inputs and progress is not None:
            progress("llm[extract_final_hs_code] request -> " + _serialize_llm_payload(system_prompt, prompt))
        payload = self.clients.chat_json(system_prompt, json.dumps(prompt, ensure_ascii=False, separators=(',', ':')))
        raw_code = payload.get(scheme.final_code_label, payload.get("hs_code", ""))
        return str(scheme.normalize_code(raw_code)) if callable(scheme.normalize_code) else _normalize_hs_code(raw_code)

    def _run_single_pass(
        self,
        query: str,
        branch_top_k: int,
        max_depth: int,
        global_top_k: int = 3,
        candidate_field_mode: CandidateFieldMode = "full",
        avoid_codes: list[str] | None = None,
        avoid_node_ids: list[str] | None = None,
        precomputed_retrieval_query: str | None = None,
        precomputed_anchor_terms: list[str] | None = None,
        precomputed_constraint_terms: list[str] | None = None,
        print_candidate_packages: bool = False,
        print_llm_inputs: bool = False,
        print_similarity_trace: bool = False,
        print_answer_trace: bool = False,
        progress: Callable[[str], None] | None = None,
        expand_from_root: bool = False,
    ) -> SearchPassResult:
        """执行单轮树搜索。

        搜索策略（改进版）：
          1. 第一层（depth=0）：从根节点出发，使用 global 全局语义检索
             获取整棵树中与 query 最相关的 top-K 节点，通过 LLM 选择
             最佳节点作为 RegTree 展开的起点。
          2. 后续层（depth>=1）：沿 RegTree 树结构逐层向下展开子节点，
             不再使用 global 跨分支检索，保证搜索路径沿树结构有序下钻。
        不使用关键词召回。
        """
        self._pass_evidence_cache.clear()
        candidate_field_mode = _normalize_candidate_field_mode(candidate_field_mode)
        if progress is not None:
            progress(f"starting single-pass tree search: query={query}")
        if precomputed_retrieval_query is not None and precomputed_anchor_terms is not None:
            retrieval_query = precomputed_retrieval_query
            anchor_terms = precomputed_anchor_terms
            constraint_terms = precomputed_constraint_terms or []
        else:
            anchor_payload = self._llm_extract_anchor_terms_with_trace(
                query,
                progress=progress,
                print_llm_inputs=print_llm_inputs,
            )
            retrieval_query = str(anchor_payload.get("retrieval_query", "")).strip() or query.strip()
            anchor_terms = _normalize_text_list(anchor_payload.get("anchor_terms", []), max_items=8)
            constraint_terms = _normalize_text_list(anchor_payload.get("constraint_terms", []), max_items=8)
            if not anchor_terms:
                anchor_terms = [retrieval_query]
        # 合并锚点和限定词，用于答案验证的回退词
        merged_terms = _merge_term_lists(anchor_terms, constraint_terms)
        core_terms = anchor_terms
        if progress is not None:
            progress(f"LLM retrieval query={retrieval_query}")
            progress(f"LLM anchor terms: {', '.join(anchor_terms)}")
            if constraint_terms:
                progress(f"LLM constraints: {', '.join(constraint_terms)}")
        embedding_query = retrieval_query
        query_vector = self._embed_query(embedding_query, progress=progress)
        query_norm = float(np.linalg.norm(query_vector))
        avoid_hit_ids = self._resolve_avoid_hit_ids(avoid_codes, avoid_node_ids)
        if progress is not None and avoid_codes:
            progress(f"avoid codes: {avoid_codes}")
        if progress is not None and avoid_node_ids:
            progress(f"avoid nodes: {avoid_node_ids}")
        avoid_hit_ids = self._resolve_avoid_hit_ids(avoid_codes, avoid_node_ids)
        if global_top_k > 0 and self.global_recall_indices:
            precomputed_global_sims = _cosine_similarity_with_norms(
                query_vector,
                self.node_vectors[self.global_recall_indices],
                query_norm=query_norm,
                matrix_norm=self.node_vector_norms[self.global_recall_indices],
            )
        else:
            precomputed_global_sims = None

        current_id = self.payload["root_id"]
        path: list[dict[str, Any]] = []
        visited_node_ids: list[str] = []
        last_candidates: list[dict[str, Any]] = []
        llm_stop = False

        for depth in range(max_depth):
            if llm_stop:
                break
            node = self.nodes[current_id]
            if not node["children"]:
                break

            if depth == 0:
                if expand_from_root:
                    candidates = self._top_child_candidates(
                        current_id,
                        query,
                        query_vector,
                        query_norm,
                        anchor_terms=anchor_terms,
                        avoid_codes=avoid_codes,
                        avoid_hit_ids=avoid_hit_ids,
                        top_k=branch_top_k,
                        global_top_k=0,
                        visited_node_ids=visited_node_ids,
                        precomputed_global_sims=precomputed_global_sims,
                    )
                    if progress is not None:
                        progress(
                            f"depth=1: [消融: 从根节点展开, global关闭]: "
                            f"top candidates ({len(candidates)} total) -> "
                            f"{_format_candidate_brief(candidates, limit=max(3, branch_top_k))}"
                        )
                else:
                    global_candidates = self._top_global_candidates(
                        query_vector=query_vector,
                        query_norm=query_norm,
                        exclude_ids={current_id, *visited_node_ids},
                        avoid_codes=avoid_codes,
                        avoid_hit_ids=avoid_hit_ids,
                        top_k=global_top_k,
                        precomputed_global_sims=precomputed_global_sims,
                    )
                    candidates = []
                    for item in global_candidates:
                        g_node_id = item["id"]
                        g_node = self.nodes[g_node_id]
                        evidence = self._candidate_evidence(
                            g_node_id,
                            query_vector,
                            query_norm=query_norm,
                            limit=1,
                            excerpt_terms=anchor_terms,
                        )
                        candidates.append(
                            {
                                "id": g_node_id,
                                "code": g_node["code"],
                                "title": g_node["title"],
                                "node_type": g_node["node_type"],
                                "score": float(item["semantic_score"]),
                                "retrieval_score": float(item["semantic_score"]),
                                "semantic_score": float(item["semantic_score"]),
                                "node_score": item.get("node_score"),
                                "retrieval_source": "global",
                                "notes": g_node["notes"][:3],
                                "exclusions": g_node["exclusions"][:3],
                                "definitions": g_node["definitions"][:3],
                                "text": _truncate(g_node["text"], 900),
                                "evidence": evidence,
                            }
                        )
                    if not candidates:
                        if progress is not None:
                            progress("depth=1: global retrieval empty, falling back to root children expansion")
                        candidates = self._top_child_candidates(
                            current_id,
                            query,
                            query_vector,
                            query_norm,
                            anchor_terms=anchor_terms,
                            avoid_codes=avoid_codes,
                            avoid_hit_ids=avoid_hit_ids,
                            top_k=branch_top_k,
                            global_top_k=0,
                            visited_node_ids=visited_node_ids,
                            precomputed_global_sims=precomputed_global_sims,
                        )
                    if progress is not None:
                        progress(
                            f"depth=1: [global pure retrieval, skip root children expansion]: "
                            f"top candidates ({len(candidates)} total) -> "
                            f"{_format_candidate_brief(candidates, limit=max(3, global_top_k))}"
                        )
            else:
                candidates = self._top_child_candidates(
                    current_id,
                    query,
                    query_vector,
                    query_norm,
                    anchor_terms=anchor_terms,
                    avoid_codes=avoid_codes,
                    avoid_hit_ids=avoid_hit_ids,
                    top_k=branch_top_k,
                    global_top_k=0,
                    visited_node_ids=visited_node_ids,
                    precomputed_global_sims=precomputed_global_sims,
                )
                if progress is not None:
                    candidate_brief_limit = max(3, branch_top_k)
                    progress(
                        f"depth={depth + 1}: [global关闭, 仅RegTree展开]: "
                        f"top candidates ({len(candidates)} total, showing {min(len(candidates), candidate_brief_limit)}) -> "
                        f"{_format_candidate_brief(candidates, limit=candidate_brief_limit)}"
                    )

            if not candidates:
                break

            if print_candidate_packages:
                self._emit_candidate_packages(
                    parent_id=current_id,
                    candidates=candidates,
                    depth=depth + 1,
                    round_query=query,
                    progress=progress,
                )
            if print_similarity_trace:
                self._emit_similarity_trace(
                    parent_id=current_id,
                    retrieval_query=retrieval_query,
                    query_vector=query_vector,
                    candidates=candidates,
                    depth=depth + 1,
                    progress=progress,
                )

            if len(candidates) == 1:
                choice = {
                    "selected_id": candidates[0]["id"],
                    "alternate_ids": [],
                    "candidate_scores": [{"id": candidates[0]["id"], "match_score": candidates[0]["score"]}],
                    "stop": False,
                    "confidence": min(1.0, candidates[0]["score"]),
                "reason": "sole candidate, auto-selected",
                }
            else:
                choice = self._llm_select(
                    query=query,
                    parent_node=self.nodes[current_id],
                    candidates=candidates,
                    candidate_field_mode=candidate_field_mode,
                    avoid_codes=avoid_codes,
                    avoid_node_ids=avoid_node_ids,
                    avoid_hit_ids=avoid_hit_ids,
                    depth=depth,
                    progress=progress,
                    print_llm_inputs=print_llm_inputs,
                    print_answer_trace=print_answer_trace,
                )
            selected_id = self._coerce_selected_id(choice, candidates)
            llm_match_scores = self._coerce_candidate_match_scores(choice, candidates)

            selection_confidence = self._coerce_selection_confidence(choice)
            selected_match_score = llm_match_scores.get(selected_id, 0.0)
            selected_reason = str(choice.get("reason", "")).strip()
            selected_child = self.nodes[selected_id]
            selected_has_children = bool(selected_child["children"])
            candidates_snapshot = [
                {
                    "id": c["id"],
                    "code": c["code"],
                    "title": c["title"],
                    "node_type": c["node_type"],
                    "retrieval_score": round(float(c.get("retrieval_score", c["score"])), 4),
                    "retrieval_source": c.get("retrieval_source", "child"),
                    "llm_match_score": round(float(llm_match_scores.get(c["id"], 0.0)), 4),
                }
                for c in candidates
            ]
            selected_candidate = next(item for item in candidates if item["id"] == selected_id)
            llm_stop = bool(choice.get("stop", False))

            path.append(
                {
                    "layer_name": selected_child["node_type"],
                    "code": selected_child["code"],
                    "title": selected_child["title"],
                    "retrieval_source": selected_candidate.get("retrieval_source", "child"),
                    "score": round(float(selected_candidate["score"]), 6),
                    "primary": True,
                    "llm_match_score": round(float(selected_match_score), 6),
                    "selection_confidence": round(float(selection_confidence), 6),
                    "local_confidence": round(float(selection_confidence), 6),
                    "reason": selected_reason,
                    "stop": llm_stop,
                    "candidates": candidates_snapshot,
                }
            )
            visited_node_ids.append(selected_id)
            last_candidates = candidates

            if depth == 0 and not expand_from_root and selected_candidate.get("retrieval_source") == "global":
                ancestor_ids = self._ancestor_chain(selected_id)
                if ancestor_ids:
                    ancestor_path_entries: list[dict[str, Any]] = []
                    for anc_id in ancestor_ids:
                        anc_node = self.nodes[anc_id]
                        ancestor_path_entries.append({
                            "layer_name": anc_node["node_type"],
                            "code": anc_node["code"],
                            "title": anc_node["title"],
                            "retrieval_source": "ancestor",
                            "score": 0.0,
                            "primary": True,
                            "llm_match_score": 0.0,
                            "selection_confidence": 0.0,
                            "local_confidence": 0.0,
                            "reason": "",
                            "stop": False,
                            "candidates": [],
                        })
                    path[:] = ancestor_path_entries + path
                    visited_node_ids[:] = ancestor_ids + visited_node_ids
                    if progress is not None:
                        progress(
                            f"depth=1: prepended {len(ancestor_ids)} ancestors -> "
                            + " -> ".join(self.nodes[aid]["code"] for aid in ancestor_ids)
                        )

            if progress is not None:
                progress(
                    f"depth={depth + 1} -> {_format_node_label(selected_child)}; "
                    f"source={selected_candidate.get('retrieval_source', 'child')}; "
                    f"match={selected_match_score:.3f}; "
                    f"selection_confidence={selection_confidence:.3f}; "
                    f"stop={llm_stop or not selected_has_children}; reason={selected_reason}"
                )

            current_id = selected_id

        final_node = self.nodes[current_id]
        alternatives = [
            {
                "code": self.nodes[item["id"]]["code"],
                "title": self.nodes[item["id"]]["title"],
                "score": round(item["score"], 6),
            }
            for item in last_candidates
            if item["id"] != current_id
        ][:3]
        node_ids_for_evidence = visited_node_ids or [current_id]
        evidence = self._collect_node_evidence(node_ids_for_evidence, query_vector, query_norm)
        retrieved_nodes = [self._node_brief(n_id) for n_id in node_ids_for_evidence]
        if path:
            last_step = path[-1]
            estimated_confidence = min(
                float(last_step.get("llm_match_score", 0.0)),
                float(last_step.get("selection_confidence", last_step.get("local_confidence", 0.0))),
            )
        else:
            estimated_confidence = 0.0
        answer_payload = {
            "answer": "",
            "confidence": estimated_confidence,
            "used_chunk_ids": [],
        }
        if progress is not None:
            progress(
                "single-pass completed: "
                f"final_node={_format_node_label(final_node)}, "
                f"evidence={len(evidence)}, "
                f"confidence={estimated_confidence}"
            )

        return SearchPassResult(
            query=query,
            retrieval_query=retrieval_query,
            final_node_id=current_id,
            final_node={
                "code": final_node["code"],
                "title": final_node["title"],
                "node_type": final_node["node_type"],
            },
            search_path=path,
            retrieved_nodes=retrieved_nodes,
            alternatives=alternatives,
            evidence=evidence,
            answer_payload=answer_payload,
        )




    def _merge_retrieved_nodes(
        self,
        passes: list[SearchPassResult],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in passes:
            pass_weight = self._pass_aggregation_weight(item)
            for position, node in enumerate(item.retrieved_nodes):
                merge_score = pass_weight - position * 0.03
                existing = merged.get(node["id"])
                if existing is None or merge_score > float(existing["_merge_score"]):
                    merged[node["id"]] = {
                        **node,
                        "_merge_score": merge_score,
                    }
        ranked = sorted(
            merged.values(),
            key=lambda item: (float(item["_merge_score"]), len(str(item.get("code", "")))),
            reverse=True,
        )
        return [{k: v for k, v in item.items() if k != "_merge_score"} for item in ranked[:limit]]

    def _merge_evidence(self, passes: list[SearchPassResult], limit: int = 10) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in passes:
            pass_weight = self._pass_aggregation_weight(item)
            for evidence in item.evidence:
                merge_score = pass_weight + float(evidence["score"])
                existing = merged.get(evidence["chunk_id"])
                if existing is None or merge_score > float(existing["_merge_score"]):
                    merged[evidence["chunk_id"]] = {
                        **evidence,
                        "_merge_score": merge_score,
                    }
        ranked = sorted(merged.values(), key=lambda item: float(item["_merge_score"]), reverse=True)
        return [{k: v for k, v in item.items() if k != "_merge_score"} for item in ranked[:limit]]

    def _emit_aggregation_trace(
        self,
        *,
        passes: list[SearchPassResult],
        best_pass: SearchPassResult,
        ordered_passes: list[SearchPassResult],
        aggregated_nodes: list[dict[str, Any]],
        aggregated_evidence: list[dict[str, Any]],
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if progress is None:
            return
        payload = {
            "best_round": passes.index(best_pass) + 1,
            "ordered_rounds": [passes.index(item) + 1 for item in ordered_passes],
            "aggregated_nodes": [
                {
                    "id": node["id"],
                    "code": node.get("code", ""),
                    "title": node.get("title", ""),
                    "node_type": node.get("node_type", ""),
                    "text": node.get("text", ""),
                }
                for node in aggregated_nodes
            ],
            "aggregated_evidence": [
                {
                    "chunk_id": item["chunk_id"],
                    "title": item.get("title", ""),
                    "pages": item.get("pages", ""),
                    "score": item.get("score", 0.0),
                    "excerpt": item.get("excerpt", ""),
                }
                for item in aggregated_evidence
            ],
        }
        progress("aggregation detail -> " + _serialize_json_payload(payload))

    def _pick_best_pass(self, passes: list[SearchPassResult]) -> SearchPassResult:
        return max(passes, key=self._pass_rank_tuple)

    def _pass_aggregation_weight(self, item: SearchPassResult) -> float:
        answer_payload = item.answer_payload
        confidence = min(_confidence_value(answer_payload.get("confidence")), 0.85)
        return (
            1.0
            + confidence * 0.5
        )

    def _pass_aggregation_rank_tuple(self, item: SearchPassResult) -> tuple[float, int, int, int]:
        final_code_len = len("".join(ch for ch in str(item.final_node.get("code", "")) if ch.isdigit()))
        return (
            self._pass_aggregation_weight(item),
            final_code_len,
            -self._generic_other_penalty(item.final_node),
            len(item.evidence),
        )

    def _pass_rank_tuple(self, item: SearchPassResult) -> tuple[int, float, int, int, int, int]:
        answer_payload = item.answer_payload
        confidence = _confidence_value(answer_payload.get("confidence"))
        final_code_len = len("".join(ch for ch in str(item.final_node.get("code", "")) if ch.isdigit()))
        leaf_depth = len(item.search_path)
        evidence_count = len(item.evidence)
        return (
            1,
            confidence,
            final_code_len,
            -self._generic_other_penalty(item.final_node),
            leaf_depth,
            evidence_count,
        )

    def _pick_aligned_pass(
        self,
        passes: list[SearchPassResult],
        fallback: SearchPassResult,
        final_answer_code: str,
    ) -> SearchPassResult:
        answer_code = _normalize_hs_code(final_answer_code)
        if not answer_code:
            return fallback

        def pass_contains_code(item: SearchPassResult) -> bool:
            if _normalize_hs_code(item.final_node.get("code", "")) == answer_code:
                return True
            for node in item.retrieved_nodes:
                if _normalize_hs_code(node.get("code", "")) == answer_code:
                    return True
            for node in item.search_path:
                if _normalize_hs_code(node.get("code", "")) == answer_code:
                    return True
            return False

        for item in reversed(passes):
            if pass_contains_code(item):
                return item
        return fallback

    def search(
        self,
        query: str,
        branch_top_k: int = 5,
        max_depth: int = 8,
        max_rounds: int = 5,
        max_stagnation_rounds: int = 2,
        global_top_k: int = 3,
        candidate_field_mode: CandidateFieldMode = "full",
        print_candidate_packages: bool = False,
        print_llm_inputs: bool = False,
        print_similarity_trace: bool = False,
        print_answer_trace: bool = False,
        print_aggregation_trace: bool = False,
        progress: Callable[[str], None] | None = None,
        expand_from_root: bool = False,
    ) -> dict[str, Any]:
        normalized_query = query.strip()
        candidate_field_mode = _normalize_candidate_field_mode(candidate_field_mode)
        passes: list[SearchPassResult] = []
        round_traces: list[dict[str, Any]] = []
        seen_query_states: set[tuple[str, str, tuple[str, ...]]] = set()
        seen_node_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()
        stagnation_rounds = 0
        current_query = normalized_query
        avoid_node_ids: list[str] = []
        visited_path_node_ids: list[str] = []
        stop_reason = "reached max search rounds"
        if expand_from_root:
            early_stop_confidence_threshold = 0.92
        else:
            early_stop_confidence_threshold = 0.85

        if progress is not None:
            progress(
                f"starting search: query={normalized_query}; "
                f"branch_top_k={branch_top_k}; global_top_k={global_top_k}; "
                f"candidate_field_mode={candidate_field_mode}; "
                f"max_depth={max_depth}; max_rounds={max_rounds}"
            )

        seen_query_states.add(
            (
                normalized_query,
                "refine_query",
                tuple(),
            )
        )

        shared_anchor_payload = self._llm_extract_anchor_terms_with_trace(
            normalized_query,
            progress=progress,
            print_llm_inputs=print_llm_inputs,
        )
        shared_retrieval_query = str(shared_anchor_payload.get("retrieval_query", "")).strip() or normalized_query.strip()
        shared_anchor_terms = _normalize_text_list(shared_anchor_payload.get("anchor_terms", []), max_items=8)
        shared_constraint_terms = _normalize_text_list(shared_anchor_payload.get("constraint_terms", []), max_items=8)
        if not shared_anchor_terms:
            shared_anchor_terms = [shared_retrieval_query]
        if progress is not None:
            progress(f"shared anchor extraction: retrieval_query={shared_retrieval_query}, anchor_terms={', '.join(shared_anchor_terms)}")
            if shared_constraint_terms:
                progress(f"shared constraints: constraint_terms={', '.join(shared_constraint_terms)}")

        current_anchor_terms = list(shared_anchor_terms)
        for round_index in range(max(1, max_rounds)):
            if progress is not None:
                progress(f"round {round_index + 1}/{max(1, max_rounds)}: query={current_query}")
            result = self._run_single_pass(
                current_query,
                branch_top_k=branch_top_k,
                max_depth=max_depth,
                global_top_k=global_top_k,
                candidate_field_mode=candidate_field_mode,
                avoid_node_ids=avoid_node_ids,
                precomputed_retrieval_query=shared_retrieval_query,
                precomputed_anchor_terms=current_anchor_terms,
                precomputed_constraint_terms=shared_constraint_terms,
                print_candidate_packages=print_candidate_packages,
                print_llm_inputs=print_llm_inputs,
                print_similarity_trace=print_similarity_trace,
                print_answer_trace=print_answer_trace,
                progress=progress,
                expand_from_root=expand_from_root,
            )
            passes.append(result)

            round_node_ids = {node["id"] for node in result.retrieved_nodes}
            round_chunk_ids = {item["chunk_id"] for item in result.evidence}
            new_node_ids = sorted(round_node_ids - seen_node_ids)
            new_chunk_ids = sorted(round_chunk_ids - seen_chunk_ids)
            seen_node_ids.update(round_node_ids)
            seen_chunk_ids.update(round_chunk_ids)
            visited_path_node_ids.extend(
                node_id
                for node_id in [step_node.get("id", "") for step_node in result.retrieved_nodes]
                if node_id and node_id not in visited_path_node_ids
            )
            if result.final_node_id and result.final_node_id not in visited_path_node_ids:
                visited_path_node_ids.append(result.final_node_id)

            made_progress = bool(new_node_ids or new_chunk_ids)
            stagnation_rounds = 0 if made_progress else stagnation_rounds + 1
            if print_answer_trace and progress is not None:
                progress(
                    f"progress[{round_index + 1}] formula -> "
                    + _serialize_json_payload(
                        {
                            "formula": "Prog^(r)=I(Delta V^(r)!=empty or Delta Z^(r)!=empty)",
                            "r": round_index + 1,
                            "Delta V^(r)": new_node_ids,
                            "Delta Z^(r)": new_chunk_ids,
                            "Prog^(r)": int(made_progress),
                        }
                    )
                )

            round_trace = {
                "round": round_index + 1,
                "query": current_query,
                "retrieval_query": result.retrieval_query,
                "answer": result.answer_payload.get("answer", ""),
                "confidence": result.answer_payload.get(
                    "confidence",
                    result.search_path[-1]["score"] if result.search_path else None,
                ),
                "final_node_id": result.final_node_id,
                "final_node": result.final_node,
                "search_path": result.search_path,
                "retrieved_nodes": result.retrieved_nodes,
                "alternatives": result.alternatives,
                "evidence": result.evidence,
                "used_chunk_ids": result.answer_payload.get("used_chunk_ids", []),
                "new_node_ids": new_node_ids,
                "new_chunk_ids": new_chunk_ids,
                "progress_flag": int(made_progress),
                "planner": None,
            }
            round_traces.append(round_trace)
            if progress is not None:
                progress(
                    f"round {round_index + 1} result: "
                    f"new_nodes={len(new_node_ids)}, new_evidence={len(new_chunk_ids)}, "
                    f"final_node={_format_node_label(result.final_node)}"
                )

            round_confidence = _confidence_value(result.answer_payload.get("confidence"))
            if round_confidence >= early_stop_confidence_threshold:
                stop_reason = (
                    f"round {round_index + 1} reached high-confidence result"
                )
                if progress is not None:
                    progress(stop_reason)
                break

            final_path_step = result.search_path[-1] if result.search_path else {}
            final_node_type = str(result.final_node.get("node_type", ""))
            final_code = "".join(ch for ch in str(result.final_node.get("code", "")) if ch.isdigit())
            final_selection_confidence = float(
                final_path_step.get(
                    "selection_confidence",
                    final_path_step.get("local_confidence", 0.0),
                )
                or 0.0
            )
            final_is_terminal_choice = bool(final_path_step.get("stop", False))
            if (
                final_is_terminal_choice
                and final_selection_confidence >= 0.75
                and (
                    (
                        final_node_type in scheme.fine_grained_node_types
                        and len(final_code) >= scheme.fine_grained_code_min_length
                    )
                    or (
                        final_node_type in scheme.coarse_grained_node_types
                        and len(final_code) >= scheme.coarse_grained_code_min_length
                    )
                )
            ):
                stop_reason = f"round {round_index + 1} hit high-match terminal node, stopping search"
                if progress is not None:
                    progress(stop_reason)
                break

            if round_index + 1 >= max(1, max_rounds):
                stop_reason = "reached max search rounds"
                break
            if stagnation_rounds >= max(1, max_stagnation_rounds):
                stop_reason = f"{stagnation_rounds} consecutive rounds with no new evidence"
                break

            explored_rounds = [
                {
                    "round": item["round"],
                    "query": item["query"],
                    "answer_summary": _format_node_label(item["final_node"]) if item.get("final_node") else "",
                    "confidence": item["confidence"],
                    "final_node": item["final_node"],
                    "final_node_id": item.get("final_node_id", ""),
                    "search_path_summary": [
                        {
                            "code": step.get("code", ""),
                            "title": step.get("title", ""),
                        }
                        for step in item.get("search_path", [])
                    ],
                    "new_node_count": len(item["new_node_ids"]),
                    "new_chunk_count": len(item["new_chunk_ids"]),
                    "planner_action": (
                        str(item.get("planner", {}).get("action", "")).strip()
                        if isinstance(item.get("planner"), dict)
                        else ""
                    ),
                }
                for item in round_traces
            ]
            planner_trace = self._llm_plan_next_round(
                original_query=normalized_query,
                explored_rounds=explored_rounds,
                remaining_rounds=max_rounds - round_index - 1,
                progress=progress,
                print_llm_inputs=print_llm_inputs,
                print_answer_trace=print_answer_trace,
            )
            fallback_trace = self._rule_based_fallback_plan(
                original_query=normalized_query,
                current_query=current_query,
                result=result,
                new_node_ids=new_node_ids,
                new_chunk_ids=new_chunk_ids,
                avoid_node_ids=avoid_node_ids,
                remaining_rounds=max_rounds - round_index - 1,
            )
            if fallback_trace is not None:
                planner_trace = fallback_trace
                if progress is not None:
                    progress("planner fallback: triggered rule-based sibling/backtrack strategy")
            round_trace["planner"] = planner_trace
            planner_action = str(planner_trace.get("action", "")).strip() or (
                "stop" if not bool(planner_trace.get("continue_search", False)) else "refine_query"
            )
            if planner_action == "switch_sibling":
                avoid_node_ids = self._normalize_node_id_list(planner_trace.get("avoid_node_ids", []))
                if result.final_node_id and result.final_node_id not in avoid_node_ids:
                    avoid_node_ids.append(result.final_node_id)
            else:
                avoid_node_ids = list(dict.fromkeys(
                    visited_path_node_ids
                    + self._normalize_node_id_list(planner_trace.get("avoid_node_ids", []))
                ))
            if progress is not None:
                progress(
                    f"planner: action={planner_action}; continue={bool(planner_trace.get('continue_search', False))}; "
                    f"focus={planner_trace.get('focus', '')}; "
                    f"reason={planner_trace.get('reason', '')}"
                )
                if avoid_node_ids:
                    progress(
                        "planner avoidance: "
                        f"avoid_nodes={avoid_node_ids or []}"
                    )

            if planner_action == "stop" or not bool(planner_trace.get("continue_search", False)):
                stop_reason = str(planner_trace.get("reason", "")).strip() or "planner determined current evidence is sufficient"
                break

            planner_focus_terms = _normalize_text_list(planner_trace.get("focus_terms", []), max_items=6, scheme=scheme)
            planner_focus_terms = [
                cleaned
                for term in planner_focus_terms
                if (cleaned := _strip_unverified_code_phrases(term, allowed_codes=_extract_query_codes(normalized_query, scheme), scheme=scheme))
            ]
            if planner_focus_terms:
                current_anchor_terms = _merge_term_lists(shared_anchor_terms, planner_focus_terms, max_items=10, scheme=scheme)
                if progress is not None:
                    progress(f"anchor enrichment: focus_terms={planner_focus_terms} -> merged_anchors={current_anchor_terms}")
            else:
                current_anchor_terms = list(shared_anchor_terms)

            raw_next_query = str(planner_trace.get("next_query", "")).strip()
            next_query = _compose_followup_query(normalized_query, planner_trace, scheme)
            if not next_query:
                stop_reason = "planner did not provide a valid next-round query"
                break
            next_state_signature = (
                next_query,
                planner_action,
                tuple(sorted(avoid_node_ids)),
            )
            if next_state_signature in seen_query_states:
                stop_reason = "planner generated duplicate query, stopping search"
                break

            seen_query_states.add(next_state_signature)
            if progress is not None:
                if raw_next_query and raw_next_query != next_query:
                    progress(f"next round query cleanup: raw={raw_next_query} -> sanitized={next_query}")
                progress(f"next round query={next_query}")
            current_query = next_query

        best_pass = self._pick_best_pass(passes)
        if print_answer_trace and progress is not None:
            progress(
                "best-round formula -> "
                + _serialize_json_payload(
                    {
                        "formula": "r*=argmax_r(b^(r),c^(r),l^(r),o^(r),d^(r),m^(r))",
                        "round_scores": [
                            {
                                "r": index + 1,
                                "b^(r)": self._pass_rank_tuple(item)[0],
                                "c^(r)": self._pass_rank_tuple(item)[1],
                                "l^(r)": self._pass_rank_tuple(item)[2],
                                "o^(r)": self._pass_rank_tuple(item)[3],
                                "d^(r)": self._pass_rank_tuple(item)[4],
                                "m^(r)": self._pass_rank_tuple(item)[5],
                            }
                            for index, item in enumerate(passes)
                        ],
                        "comparison": "lexicographic",
                        "r*": passes.index(best_pass) + 1,
                    }
                )
            )
        ordered_passes = sorted(passes, key=self._pass_aggregation_rank_tuple, reverse=True)
        aggregated_nodes = self._merge_retrieved_nodes(ordered_passes)
        aggregated_evidence = self._merge_evidence(ordered_passes)
        if progress is not None:
            progress(
                f"starting final answer aggregation: nodes={len(aggregated_nodes)}, evidence={len(aggregated_evidence)}"
            )
        if print_aggregation_trace:
            self._emit_aggregation_trace(
                passes=passes,
                best_pass=best_pass,
                ordered_passes=ordered_passes,
                aggregated_nodes=aggregated_nodes,
                aggregated_evidence=aggregated_evidence,
                progress=progress,
            )
        best_pass_confidence = _confidence_value(best_pass.answer_payload.get("confidence"))
        if print_answer_trace and progress is not None:
            progress(
                "final-aggregation formula -> "
                + _serialize_json_payload(
                    {
                        "formula": "a_final=G_theta(x,p_agg(x),E_agg(x))",
                        "r*": passes.index(best_pass) + 1,
                        "best_round_answer_confidence": best_pass_confidence,
                        "aggregated_node_count": len(aggregated_nodes),
                        "aggregated_evidence_count": len(aggregated_evidence),
                    }
                )
            )
        if print_answer_trace and progress is not None:
            progress(
                "final-aggregation decision -> "
                + _serialize_json_payload(
                    {
                        "strategy": "run_final_answer_aggregation",
                        "chosen_round": passes.index(best_pass) + 1,
                    }
                )
            )
        final_answer_evidence = self._expand_evidence_for_answer(aggregated_evidence)
        final_answer_payload = self._llm_answer(
            query=normalized_query,
            retrieved_nodes=aggregated_nodes,
            evidence=final_answer_evidence,
            explored_rounds=[
                {
                    "round": item["round"],
                    "query": item["query"],
                    "retrieval_query": item["retrieval_query"],
                    "answer_summary": _format_node_label(item["final_node"]) if item.get("final_node") else "",
                    "confidence": item["confidence"],
                    "final_node": item["final_node"],
                }
                for item in round_traces
            ],
            progress=progress,
            print_llm_inputs=print_llm_inputs,
            print_answer_trace=print_answer_trace,
            answer_stage="final",
            expand_from_root=expand_from_root,
        )
        final_answer_payload = _enforce_answer_support(
            query=normalized_query,
            answer_payload=final_answer_payload,
            retrieved_nodes=aggregated_nodes,
            evidence=final_answer_evidence,
        )

        reasoning = str(final_answer_payload.get("reasoning", "")).strip()
        answer_text = str(final_answer_payload.get("answer", "")).strip()
        full_answer = f"{reasoning}\n\n{answer_text}".strip() if reasoning else answer_text

        conflict_info = self._detect_conflicting_paths(passes)
        if conflict_info.get("has_conflict"):
            current_confidence = _confidence_value(final_answer_payload.get("confidence"))
            dampened_confidence = min(current_confidence, max(0.2, current_confidence * 0.72))
            final_answer_payload["confidence"] = round(dampened_confidence, 4)
            conflict_note = (
                "检索路径存在分支冲突："
                + "、".join(conflict_info.get("competing_prefixes", []))
                + "。已下调最终置信度。"
            )
            if conflict_note not in full_answer:
                full_answer = f"{full_answer}\n\n{conflict_note}".strip()
            if progress is not None:
                progress("conflict detection: " + _serialize_json_payload(conflict_info))

        answer_code_from_answer = _normalize_hs_code(final_answer_payload.get("final_code", ""))
        if answer_code_from_answer:
            final_answer_code = answer_code_from_answer
        else:
            final_answer_code = self._llm_extract_final_hs_code(
                query=normalized_query,
                answer=full_answer,
                retrieved_nodes=aggregated_nodes,
                explored_rounds=[
                    {
                        "round": item["round"],
                        "answer_summary": item["answer"],
                        "final_node": item["final_node"],
                    }
                    for item in round_traces
                ],
                progress=progress,
                print_llm_inputs=print_llm_inputs,
            )
        aligned_pass = self._pick_aligned_pass(passes, best_pass, final_answer_code)
        if print_aggregation_trace and progress is not None:
            progress(
                "aggregation alignment -> "
                + _serialize_json_payload(
                    {
                        "final_answer_code": final_answer_code,
                        "aligned_round": passes.index(aligned_pass) + 1,
                        "aligned_final_node": aligned_pass.final_node,
                    }
                )
            )

        return {
            "query": normalized_query,
            "reasoning": reasoning,
            "answer": full_answer,
            "confidence": final_answer_payload.get(
                "confidence",
                best_pass.answer_payload.get("confidence"),
            ),
            "used_chunk_ids": final_answer_payload.get("used_chunk_ids", []),
            "final_node_id": aligned_pass.final_node_id,
            "final_node": aligned_pass.final_node,
            "search_path": aligned_pass.search_path,
            "retrieved_nodes": aggregated_nodes,
            "alternatives": aligned_pass.alternatives,
            "evidence": aggregated_evidence,
            "search_rounds": round_traces,
            "round_count": len(round_traces),
            "stop_reason": stop_reason,
            "conflict_info": conflict_info,
            "final_answer_code": final_answer_code,
            "best_round": next(
                (
                    item["round"]
                    for item in round_traces
                    if item["query"] == aligned_pass.query and item["final_node"] == aligned_pass.final_node
                ),
                None,
            ),
        }
