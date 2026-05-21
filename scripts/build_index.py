from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regtree_agent.config import Settings
from regtree_agent.online import OnlineClients
from regtree_agent.rules import normalize_dataset_rule_name
from regtree_agent.tree_index import BuildRuleOptions, build_and_save_index, rebuild_vectors_from_tree


def _progress(message: str) -> None:
    print(f"[build_index] {message}", file=sys.stderr, flush=True)


def _storage_path(settings: Settings, filename: str, *, dataset_name: str | None = None, dataset_path: str | None = None) -> Path:
    base_dir = settings.rag_storage_dir
    resolved_dataset_name = normalize_dataset_rule_name(dataset_name, dataset_path)
    if resolved_dataset_name:
        base_dir = base_dir / resolved_dataset_name
    return base_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tree index with configurable extraction rules")
    parser.add_argument("--rule", help="Override extraction rule for all input files")
    parser.add_argument("--rule-map", help="JSON file mapping filename patterns to rule names")
    parser.add_argument("--rule-file", help="JSON file defining extraction rule profiles")
    parser.add_argument("--dataset-name", help="Dataset name, maps to rules/<dataset-name>/")
    parser.add_argument("--dataset-path", help="Dataset file path to derive rules/<dataset_name>/")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted build from checkpoint (default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume, rebuild from scratch",
    )
    parser.add_argument(
        "--checkpoint-file",
        help="Custom checkpoint file path",
    )
    parser.add_argument(
        "--print-llm-units",
        action="store_true",
        help="Print raw and normalized LLM-extracted units",
    )
    parser.add_argument(
        "--print-llm-window-split",
        action="store_true",
        help="Print LLM window split prompt/payload and rebuilt windows",
    )
    parser.add_argument(
        "--print-llm-unit-attachments",
        action="store_true",
        help="Print how units are attached to evidence chunks",
    )
    parser.add_argument(
        "--reuse-tree",
        action="store_true",
        help="Reuse existing tree JSON, rebuild vectors only",
    )
    args = parser.parse_args()

    settings = Settings.load(ROOT)
    _progress(f"workspace={settings.workspace_root}")
    _progress(f"rules_dir={settings.rules_dir}")
    _progress(f"embedding_model={settings.embedding_model}")
    _progress(f"chat_model={settings.chat_model}")
    clients = OnlineClients(settings)
    rule_options = BuildRuleOptions(
        override_rule=args.rule,
        rule_map_path=Path(args.rule_map).resolve() if args.rule_map else None,
        rule_file=Path(args.rule_file).resolve() if args.rule_file else None,
        dataset_name=args.dataset_name,
        dataset_path=Path(args.dataset_path).resolve() if args.dataset_path else None,
        resume=False if args.no_resume else True,
        checkpoint_path=Path(args.checkpoint_file).resolve() if args.checkpoint_file else None,
        print_llm_units=args.print_llm_units,
        print_llm_window_split=args.print_llm_window_split,
        print_llm_unit_attachments=args.print_llm_unit_attachments,
    )
    if args.reuse_tree:
        tree_path, vectors_path = rebuild_vectors_from_tree(
            clients,
            tree_path=_storage_path(
                settings,
                "regtree_tree.json",
                dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
            ),
            vectors_path=_storage_path(
                settings,
                "regtree_vectors.npz",
                dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
            ),
            progress=_progress,
        )
    else:
        tree_path, vectors_path = build_and_save_index(
            settings,
            clients,
            rule_options=rule_options,
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
