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
from regtree_agent.search import RegTreeSearcher, SearchArtifacts
from regtree_agent.rules import normalize_dataset_rule_name


DEFAULT_QUERY = "生牛皮（包括水牛皮）、生马皮：未剖层的整张皮，简单干燥的每张重量不超过8千克，干盐腌的不超过10千克，鲜的、湿盐腌的或以其他方法保藏的不超过16千克的6位hs码是什么？"


def _progress(message: str) -> None:
    print(f"[query_index] {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query regtree index")
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("--dataset-name", help="Dataset name under rag-storage/")
    parser.add_argument("--dataset-path", help="Dataset file path to derive name")
    parser.add_argument("--branch-top-k", type=int, default=5, help="Top-K child candidates per branch node")
    parser.add_argument("--global-top-k", type=int, default=8, help="Top-K global retrieval candidates at depth 0")
    parser.add_argument("--max-depth", type=int, default=8, help="Max search depth per pass")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max multi-round search rounds")
    args = parser.parse_args()

    query = " ".join(args.query).strip() or DEFAULT_QUERY
    settings = Settings.load(ROOT)
    clients = OnlineClients(settings)
    dataset_name = normalize_dataset_rule_name(args.dataset_name, args.dataset_path)
    base_dir = settings.rag_storage_dir / dataset_name if dataset_name else settings.rag_storage_dir
    searcher = RegTreeSearcher(
        settings,
        clients,
        SearchArtifacts(
            tree_path=base_dir / "regtree_tree.json",
            vectors_path=base_dir / "regtree_vectors.npz",
        ),
    )
    result = searcher.search(
        query,
        progress=_progress,
        branch_top_k=args.branch_top_k,
        global_top_k=args.global_top_k,
        max_depth=args.max_depth,
        max_rounds=args.max_rounds,
    )
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
