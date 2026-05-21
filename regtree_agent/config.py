from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class Settings:
    workspace_root: Path
    input_dir: Path
    data_dir: Path
    rag_storage_dir: Path
    rules_dir: Path
    chat_base_url: str
    chat_api_key: str
    chat_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    temperature: float
    max_tokens: int

    @property
    def generated_dir(self) -> Path:
        """Backward-compatible alias for the old storage directory name."""
        return self.rag_storage_dir

    @classmethod
    def load(cls, workspace_root: str | Path | None = None) -> "Settings":
        root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
        _load_dotenv(root / ".env")
        return cls(
            workspace_root=root,
            input_dir=root / "input",
            data_dir=root / "data",
            rag_storage_dir=root / "rag-storage",
            rules_dir=root / "rules",
            chat_base_url=os.environ["OPENAI_CHAT_BASE_URL"],
            chat_api_key=os.environ["OPENAI_CHAT_API_KEY"],
            chat_model=os.environ["OPENAI_MODEL"],
            embedding_base_url=os.environ["OPENAI_EMBEDDING_BASE_URL"],
            embedding_api_key=os.environ["OPENAI_EMBEDDING_API_KEY"],
            embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4000")),
        )
