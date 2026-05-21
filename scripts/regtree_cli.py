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
from regtree_agent.search import RegTreeSearcher, SearchArtifacts
from regtree_agent.tree_index import BuildRuleOptions, build_and_save_index, rebuild_vectors_from_tree


DEFAULT_TREE_FILE = "regtree_tree.json"
DEFAULT_VECTORS_FILE = "regtree_vectors.npz"


def _progress(message: str) -> None:
    print(f"[query] {message}", file=sys.stderr, flush=True)


def _storage_path(
    settings: Settings,
    filename: str | None,
    *,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    default_filename: str,
) -> Path:
    path = Path(filename or default_filename)
    if path.is_absolute():
        return path
    base_dir = settings.rag_storage_dir
    resolved_dataset_name = normalize_dataset_rule_name(dataset_name, dataset_path)
    if resolved_dataset_name:
        base_dir = base_dir / resolved_dataset_name
    return base_dir / path


def build_command(args: argparse.Namespace) -> None:
    settings = Settings.load(ROOT)
    clients = OnlineClients(settings)
    tree_path = _storage_path(
        settings,
        args.tree_file,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        default_filename=DEFAULT_TREE_FILE,
    )
    vectors_path = _storage_path(
        settings,
        args.vectors_file,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        default_filename=DEFAULT_VECTORS_FILE,
    )
    if args.reuse_tree:
        built_tree, built_vectors = rebuild_vectors_from_tree(
            clients,
            tree_path=tree_path,
            vectors_path=vectors_path,
            progress=_progress,
        )
    else:
        built_tree, built_vectors = build_and_save_index(
            settings,
            clients,
            rule_options=BuildRuleOptions(
                override_rule=args.rule,
                rule_map_path=(ROOT / args.rule_map).resolve() if args.rule_map else None,
                rule_file=(ROOT / args.rule_file).resolve() if args.rule_file else None,
                dataset_name=args.dataset_name,
                dataset_path=(ROOT / args.dataset_path).resolve() if args.dataset_path else None,
                resume=False if args.no_resume else True,
                checkpoint_path=(ROOT / args.checkpoint_file).resolve() if args.checkpoint_file else None,
                print_llm_units=args.print_llm_units,
                print_llm_window_split=args.print_llm_window_split,
                print_llm_unit_attachments=args.print_llm_unit_attachments,
            ),
            tree_path=tree_path,
            vectors_path=vectors_path,
            progress=_progress,
        )
    result = {
        "tree_path": str(built_tree),
        "vectors_path": str(built_vectors),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _build_trace_payload(
    query: str,
    result: dict,
    searcher: RegTreeSearcher,
    args: argparse.Namespace,
) -> dict:
    search_rounds = result.get("search_rounds", [])
    anchor_terms_from_path = []
    constraint_terms_from_path = []
    if search_rounds:
        first_round = search_rounds[0]
        rq = str(first_round.get("retrieval_query", "")).strip()
        for step in first_round.get("search_path", []):
            if step.get("retrieval_source") == "global":
                anchor_terms_from_path.append(str(step.get("title", "")))

    trace_nodes = {}
    for node_id, node in searcher.nodes.items():
        trace_nodes[node_id] = {
            "id": node_id,
            "code": node.get("code", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "parent_id": searcher.parent_by_id.get(node_id, ""),
        }

    trace_rounds = []
    for rd in search_rounds:
        path_steps = rd.get("search_path", [])
        trace_steps = []
        for step in path_steps:
            raw_candidates = step.get("candidates", [])
            trace_candidates = [
                {
                    "id": c.get("id", ""),
                    "code": c.get("code", ""),
                    "title": c.get("title", ""),
                    "node_type": c.get("node_type", ""),
                    "retrieval_score": c.get("retrieval_score", 0.0),
                    "retrieval_source": c.get("retrieval_source", "child"),
                    "llm_match_score": c.get("llm_match_score", 0.0),
                }
                for c in raw_candidates
            ]
            trace_steps.append({
                "layer_name": step.get("layer_name", ""),
                "code": step.get("code", ""),
                "title": step.get("title", ""),
                "retrieval_source": step.get("retrieval_source", ""),
                "semantic_score": step.get("score", 0.0),
                "llm_match_score": step.get("llm_match_score", 0.0),
                "selection_confidence": step.get("selection_confidence", 0.0),
                "primary": step.get("primary", False),
                "stop": step.get("stop", False),
                "reason": step.get("reason", ""),
                "candidates": trace_candidates,
            })

        planner = rd.get("planner") or {}
        trace_rounds.append({
            "round": rd.get("round", 0),
            "query": rd.get("query", ""),
            "retrieval_query": rd.get("retrieval_query", ""),
            "final_node": rd.get("final_node", {}),
            "final_node_id": rd.get("final_node_id", ""),
            "search_path": trace_steps,
            "alternatives": rd.get("alternatives", []),
            "new_node_ids": rd.get("new_node_ids", []),
            "new_chunk_ids": rd.get("new_chunk_ids", []),
            "planner": {
                "action": planner.get("action", ""),
                "continue_search": planner.get("continue_search", False),
                "focus": planner.get("focus", ""),
                "focus_terms": planner.get("focus_terms", []),
                "reason": planner.get("reason", ""),
            } if planner else None,
        })

    return {
        "query": query,
        "dataset_name": args.dataset_name or "",
        "params": {
            "branch_top_k": args.branch_top_k,
            "global_top_k": args.global_top_k,
            "max_depth": args.max_depth,
            "max_rounds": args.max_rounds,
            "max_stagnation_rounds": args.max_stagnation_rounds,
            "candidate_field_mode": args.candidate_field_mode,
            "expand_from_root": args.expand_from_root,
        },
        "result": {
            "answer": result.get("answer", ""),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.0),
            "final_answer_code": result.get("final_answer_code", ""),
            "stop_reason": result.get("stop_reason", ""),
            "round_count": result.get("round_count", 0),
            "conflict_info": result.get("conflict_info", {}),
        },
        "final_node": result.get("final_node", {}),
        "search_path": result.get("search_path", []),
        "retrieved_nodes": result.get("retrieved_nodes", []),
        "evidence": [
            {
                "chunk_id": ev.get("chunk_id", ""),
                "title": ev.get("title", ""),
                "pages": ev.get("pages", ""),
                "score": ev.get("score", 0.0),
                "excerpt": ev.get("excerpt", ""),
            }
            for ev in result.get("evidence", [])
        ],
        "rounds": trace_rounds,
        "nodes_sample": dict(list(trace_nodes.items())[:500]),
    }


def query_command(args: argparse.Namespace) -> None:
    settings = Settings.load(ROOT)
    clients = OnlineClients(settings)
    tree_path = _storage_path(
        settings,
        args.tree_file,
        dataset_name=args.dataset_name,
        default_filename=DEFAULT_TREE_FILE,
    )
    vectors_path = _storage_path(
        settings,
        args.vectors_file,
        dataset_name=args.dataset_name,
        default_filename=DEFAULT_VECTORS_FILE,
    )
    searcher = RegTreeSearcher(
        settings,
        clients,
        SearchArtifacts(
            tree_path=tree_path,
            vectors_path=vectors_path,
        ),
    )
    result = searcher.search(
        args.query,
        branch_top_k=args.branch_top_k,
        max_depth=args.max_depth,
        max_rounds=args.max_rounds,
        max_stagnation_rounds=args.max_stagnation_rounds,
        global_top_k=args.global_top_k,
        candidate_field_mode=args.candidate_field_mode,
        print_candidate_packages=args.print_candidate_packages,
        print_llm_inputs=args.print_llm_inputs,
        print_similarity_trace=args.print_similarity_trace,
        print_answer_trace=args.print_answer_trace,
        print_aggregation_trace=args.print_aggregation_trace,
        progress=None if args.quiet else _progress,
        expand_from_root=args.expand_from_root,
    )

    if args.trace_output:
        trace = _build_trace_payload(args.query, result, searcher, args)
        trace_path = Path(args.trace_output)
        if not trace_path.is_absolute():
            trace_path = ROOT / trace_path
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _progress(f"trace 已保存到 {trace_path}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Online regulation tree build/query CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build tree and embedding index")
    build_parser.add_argument("--tree-file", help=f"Tree file name under rag-storage/<dataset-name>/; default {DEFAULT_TREE_FILE}")
    build_parser.add_argument("--vectors-file", help=f"Vector file name under rag-storage/<dataset-name>/; default {DEFAULT_VECTORS_FILE}")
    build_parser.add_argument("--rule", help="Override extraction rule for all input files")
    build_parser.add_argument("--rule-map", help="JSON file mapping filename patterns to rule names")
    build_parser.add_argument("--rule-file", help="JSON file defining extraction rule profiles")
    build_parser.add_argument("--dataset-name", help="Dataset-specific rules directory name under rules/")
    build_parser.add_argument("--dataset-path", help="Dataset file path used to derive rules/<dataset_name>/")
    build_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted LLM tree build from checkpoint (enabled by default)",
    )
    build_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and rebuild from scratch",
    )
    build_parser.add_argument(
        "--checkpoint-file",
        help="Optional checkpoint file path for interrupted LLM tree builds",
    )
    build_parser.add_argument(
        "--print-llm-units",
        action="store_true",
        help="Print raw and normalized LLM-extracted units during llm tree building",
    )
    build_parser.add_argument(
        "--print-llm-window-split",
        action="store_true",
        help="Print LLM window split prompt/payload and rebuilt windows during llm tree building",
    )
    build_parser.add_argument(
        "--print-llm-unit-attachments",
        action="store_true",
        help="Print how normalized units are attached to evidence chunks during llm tree building",
    )
    build_parser.add_argument(
        "--reuse-tree",
        action="store_true",
        help="Reuse the existing tree JSON and rebuild only the vectors file",
    )
    build_parser.set_defaults(func=build_command)

    query_parser = subparsers.add_parser("query", help="Query built tree index")
    query_parser.add_argument("query")
    query_parser.add_argument("--dataset-name", help="Dataset storage directory name under rag-storage/")
    query_parser.add_argument("--tree-file", help=f"Tree file name under rag-storage/<dataset-name>/; default {DEFAULT_TREE_FILE}")
    query_parser.add_argument("--vectors-file", help=f"Vector file name under rag-storage/<dataset-name>/; default {DEFAULT_VECTORS_FILE}")
    query_parser.add_argument("--branch-top-k", type=int, default=5)
    query_parser.add_argument(
        "--global-top-k",
        type=int,
        default=8,
        help="每层额外注入的全局向量召回节点数；设为 0 可关闭",
    )
    query_parser.add_argument("--max-depth", type=int, default=8)
    query_parser.add_argument("--max-rounds", type=int, default=3)
    query_parser.add_argument("--max-stagnation-rounds", type=int, default=2)
    query_parser.add_argument(
        "--candidate-field-mode",
        choices=["title_only", "title_evidence", "title_text", "full"],
        default="full",
        help="候选节点字段消融模式，默认 full",
    )
    query_parser.add_argument(
        "--print-candidate-packages",
        action="store_true",
        help="Print full candidate packages Gamma(u) for each search depth to stderr",
    )
    query_parser.add_argument(
        "--print-llm-inputs",
        action="store_true",
        help="Print the full system prompt and user payload before each chat model call",
    )
    query_parser.add_argument(
        "--print-similarity-trace",
        action="store_true",
        help="Print the local similarity formula trace for each depth, including query/node vector previews",
    )
    query_parser.add_argument(
        "--print-answer-trace",
        action="store_true",
        help="Print formula traces for multi-round history H^(r) and final answer a^(r)=G_theta(x,p^(r)(x),E^(r)(x))",
    )
    query_parser.add_argument(
        "--print-aggregation-trace",
        action="store_true",
        help="Print cross-round aggregation details, including merged nodes, merged evidence, and alignment",
    )
    query_parser.add_argument("--quiet", action="store_true", help="Hide query progress logs on stderr")
    query_parser.add_argument(
        "--expand-from-root",
        action="store_true",
        help="消融实验：从根节点展开子节点，第一层不使用全局检索",
    )
    query_parser.add_argument(
        "--trace-output",
        help="将结构化搜索 trace 导出为 JSON 文件路径（用于可视化）",
    )
    query_parser.set_defaults(func=query_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
