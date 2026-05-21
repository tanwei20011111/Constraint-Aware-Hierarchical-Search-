from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
import threading
from typing import Any, Callable

from openai import BadRequestError, OpenAI

from .prompts import (
    JSON_FALLBACK_SUFFIX,
    JSON_REPAIR_SYSTEM_PROMPT,
    JSON_REPAIR_USER_TEMPLATE,
    JSON_RETRY_SYSTEM_SUFFIX,
    JSON_RETRY_USER_TEMPLATE,
)

from .config import Settings


EMBEDDING_MAX_TEXT_LENGTH = 6000


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _extract_json_body(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def _cleanup_json_like_text(text: str) -> str:
    cleaned = _extract_json_body(_strip_code_fence(text))
    # Remove trailing commas before closing braces/brackets.
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned.strip()


def _parse_json_like_text(text: str) -> dict[str, Any]:
    cleaned = _cleanup_json_like_text(text)
    if not cleaned:
        raise json.JSONDecodeError("Empty JSON content", text, 0)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected dict JSON payload, got {type(parsed).__name__}")
    return parsed


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text_payload = item.get("text", "")
                if isinstance(text_payload, dict):
                    text = str(text_payload.get("value", "")).strip()
                else:
                    text = str(text_payload).strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _preview_text(text: str, limit: int = 300) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    finish_reason: str
    refusal: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _preview_response_meta(result: ChatCompletionResult) -> str:
    parts: list[str] = []
    if result.finish_reason:
        parts.append(f"finish_reason={result.finish_reason}")
    if result.refusal:
        parts.append(f"refusal={_preview_text(result.refusal, limit=120)}")
    if result.text:
        parts.append(f"content={_preview_text(result.text)}")
    else:
        parts.append("content=[empty response]")
    return "; ".join(parts)


def _is_empty_length_response(result: ChatCompletionResult) -> bool:
    return result.finish_reason == "length" and not result.text.strip()


def _build_json_repair_prompt(content: str) -> str:
    return JSON_REPAIR_USER_TEMPLATE.format(content=content)


def _build_json_retry_prompt(user_prompt: str, invalid_content: str, error: Exception) -> str:
    invalid_preview = _preview_text(invalid_content) if invalid_content.strip() else "[empty response]"
    return JSON_RETRY_USER_TEMPLATE.format(
        invalid_preview=invalid_preview,
        error_type=type(error).__name__,
        error=error,
        user_prompt=user_prompt,
    )


class OnlineClients:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat_supports_response_format: bool | None = None
        self.chat_client = OpenAI(
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
        )
        self.embedding_client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        self._usage_lock = threading.Lock()
        self._usage = {
            "chat_prompt_tokens": 0,
            "chat_completion_tokens": 0,
            "chat_total_tokens": 0,
            "embedding_prompt_tokens": 0,
            "embedding_total_tokens": 0,
        }

    def usage_snapshot(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def _add_usage(self, **values: int) -> None:
        with self._usage_lock:
            for key, value in values.items():
                self._usage[key] = int(self._usage.get(key, 0)) + int(value or 0)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 8,
        progress: Callable[[str], None] | None = None,
        label: str = "embedding",
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        total = len(texts)
        total_batches = (total + batch_size - 1) // batch_size if total else 0
        for batch_index, start in enumerate(range(0, total, batch_size), start=1):
            raw_batch = texts[start : start + batch_size]
            batch = [self._prepare_embedding_text(text) for text in raw_batch]
            if progress is not None:
                progress(
                    f"{label}: batch {batch_index}/{total_batches} "
                    f"({start + 1}-{start + len(batch)}/{total})"
                )
                truncated = sum(1 for original, prepared in zip(raw_batch, batch, strict=True) if original != prepared)
                if truncated:
                    progress(f"{label}: truncated {truncated} texts in batch {batch_index} to <= {EMBEDDING_MAX_TEXT_LENGTH} chars")
            response = self.embedding_client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                self._add_usage(
                    embedding_prompt_tokens=prompt_tokens,
                    embedding_total_tokens=total_tokens,
                )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def _prepare_embedding_text(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        if not cleaned:
            return "(empty)"
        if len(cleaned) > EMBEDDING_MAX_TEXT_LENGTH:
            return cleaned[:EMBEDDING_MAX_TEXT_LENGTH]
        return cleaned

    def _chat_completion_result(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        request_kwargs: dict[str, Any] = {
            "model": self.settings.chat_model,
            "temperature": temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + " /no_think"},
            ],
        }
        use_response_format = (
            response_format is not None and self._chat_supports_response_format is not False
        )
        if use_response_format:
            request_kwargs["response_format"] = response_format

        try:
            response = self.chat_client.chat.completions.create(**request_kwargs)
            if use_response_format:
                self._chat_supports_response_format = True
        except BadRequestError as exc:
            if not use_response_format or not self._is_unsupported_response_format_error(exc):
                raise
            self._chat_supports_response_format = False
            fallback_kwargs = dict(request_kwargs)
            fallback_kwargs.pop("response_format", None)
            response = self.chat_client.chat.completions.create(**fallback_kwargs)

        choice = response.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", "") or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
        self._add_usage(
            chat_prompt_tokens=prompt_tokens,
            chat_completion_tokens=completion_tokens,
            chat_total_tokens=total_tokens,
        )
        return ChatCompletionResult(
            text=_message_content_to_text(message.content),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            refusal=str(refusal).strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _is_unsupported_response_format_error(exc: BadRequestError) -> bool:
        message = str(exc).lower()
        return "response_format" in message and "support" in message

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        errors: list[str] = []
        initial_result = self._chat_completion_result(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.settings.temperature,
            response_format={"type": "json_object"},
        )
        content = initial_result.text
        try:
            return _parse_json_like_text(content)
        except Exception as exc:
            errors.append(
                f"initial response parse failed: {type(exc).__name__}: {exc}; "
                f"{_preview_response_meta(initial_result)}"
            )
            if content.strip():
                repaired = ""
                repair_result = ChatCompletionResult(text="", finish_reason="", refusal="")
                try:
                    repair_result = self._chat_completion_result(
                        system_prompt=JSON_REPAIR_SYSTEM_PROMPT,
                        user_prompt=_build_json_repair_prompt(content),
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    repaired = repair_result.text
                    return _parse_json_like_text(repaired)
                except Exception as repair_exc:
                    errors.append(
                        f"repair response parse failed: {type(repair_exc).__name__}: {repair_exc}; "
                        f"{_preview_response_meta(repair_result)}"
                    )
            retry_prompt = _build_json_retry_prompt(user_prompt, content, exc)

        retry_system_prompt = system_prompt + JSON_RETRY_SYSTEM_SUFFIX
        retry_result = self._chat_completion_result(
            system_prompt=retry_system_prompt,
            user_prompt=retry_prompt,
            temperature=0,
            response_format={"type": "json_object"},
        )
        retry_content = retry_result.text
        try:
            return _parse_json_like_text(retry_content)
        except Exception as retry_exc:
            errors.append(
                f"retry response parse failed: {type(retry_exc).__name__}: {retry_exc}; "
                f"{_preview_response_meta(retry_result)}"
            )
            if retry_content.strip():
                repaired_retry = ""
                retry_repair_result = ChatCompletionResult(text="", finish_reason="", refusal="")
                try:
                    retry_repair_result = self._chat_completion_result(
                        system_prompt=JSON_REPAIR_SYSTEM_PROMPT,
                        user_prompt=_build_json_repair_prompt(retry_content),
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    repaired_retry = retry_repair_result.text
                    return _parse_json_like_text(repaired_retry)
                except Exception as repair_retry_exc:
                    errors.append(
                        f"retry repair parse failed: {type(repair_retry_exc).__name__}: {repair_retry_exc}; "
                        f"{_preview_response_meta(retry_repair_result)}"
                    )
            if _is_empty_length_response(initial_result) or _is_empty_length_response(retry_result):
                fallback_result = ChatCompletionResult(text="", finish_reason="", refusal="")
                try:
                    self._chat_supports_response_format = False
                    fallback_system_prompt = retry_system_prompt + JSON_FALLBACK_SUFFIX
                    fallback_result = self._chat_completion_result(
                        system_prompt=fallback_system_prompt,
                        user_prompt=retry_prompt,
                        temperature=0,
                        response_format=None,
                    )
                    return _parse_json_like_text(fallback_result.text)
                except Exception as fallback_exc:
                    errors.append(
                        f"non-json-mode fallback parse failed: {type(fallback_exc).__name__}: {fallback_exc}; "
                        f"{_preview_response_meta(fallback_result)}"
                    )
            raise ValueError("Model did not return valid JSON. " + " | ".join(errors)) from retry_exc
