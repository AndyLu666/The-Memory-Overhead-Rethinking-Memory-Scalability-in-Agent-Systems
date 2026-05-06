from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from shutil import copyfile

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

pd = None
Config = None
GraphRAG = None
update_logger_path = None
logger = None
Evaluator = None
FinalReportGenerator = None
RAGQueryDataset = None
aggregate_memos_metrics = None
derive_memos_metrics = None
sanitize_trace_record_for_export = None


def _load_runtime_dependencies() -> None:
    global pd
    global Config
    global GraphRAG
    global update_logger_path
    global logger
    global Evaluator
    global FinalReportGenerator
    global RAGQueryDataset
    global aggregate_memos_metrics
    global derive_memos_metrics
    global sanitize_trace_record_for_export
    if Config is not None:
        return

    import pandas as _pd
    from init.config import Config as _Config
    from init.graph_rag import GraphRAG as _GraphRAG
    from init.logger import update_logger_path as _update_logger_path, logger as _logger
    from evaluation.evaluator import Evaluator as _Evaluator
    from utils.final_report import FinalReportGenerator as _FinalReportGenerator
    from data.query_dataset import RAGQueryDataset as _RAGQueryDataset
    from memos_stats import (
        aggregate_memos_metrics as _aggregate_memos_metrics,
        derive_memos_metrics as _derive_memos_metrics,
        sanitize_trace_record_for_export as _sanitize_trace_record_for_export,
    )

    pd = _pd
    Config = _Config
    GraphRAG = _GraphRAG
    update_logger_path = _update_logger_path
    logger = _logger
    Evaluator = _Evaluator
    FinalReportGenerator = _FinalReportGenerator
    RAGQueryDataset = _RAGQueryDataset
    aggregate_memos_metrics = _aggregate_memos_metrics
    derive_memos_metrics = _derive_memos_metrics
    sanitize_trace_record_for_export = _sanitize_trace_record_for_export

def _sanitize_for_json(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            key = str(k)
            key_lower = key.lower()
            if "embedding" in key_lower or key_lower in ("vector", "vectors"):
                continue
            cleaned[key] = _sanitize_for_json(v)
        return cleaned
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if hasattr(obj, "__fspath__"):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _is_query_error_output(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    return (
        t.startswith("Error processing query:")
        or "react_agent_action_parse_failed" in t
        or "react_agent_missing_finish_answer" in t
        or "react_agent_invalid_action" in t
        or "react_agent_missing_retrieve_query" in t
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic Memory Graph RAG System")
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("-external_graph", type=str, help="Path to external tree file to load from.")
    parser.add_argument("-root", type=str, default="", help="Root directory to prefix result/config/metric paths.")
    parser.add_argument("-query", type=str, default=None, help="Whether to run query and evaluation (1 to enable, 0 to disable). If not specified, will prompt user.")
    return parser.parse_args()

def check_dirs(config: Config, root: str):
    """Create necessary directories based on root name."""
    # Create root folder under results directory
    if root:
        base_dir = os.path.join("./results", root)
    else:
        base_dir = "./results"

    # Only create results subdirectory
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    
    # Also create the base directory for pkl and log files
    os.makedirs(base_dir, exist_ok=True)

    return base_dir, result_dir

async def process_queries_async(query_dataset, graph_rag, dataset_len, config):
    """Async function to process all queries with real-time evaluation."""
    _load_runtime_dependencies()
    from evaluation.llm_evaluator import LLMEvaluator
    from evaluation.session_matching_evaluator import SessionMatchingEvaluator
    from evaluation.evaluator import Evaluator
    from init.logger import logger
    
    all_res = []
    
    # Initialize evaluators
    enable_llm_eval = config.evaluation.enable_llm_eval if config else False
    if enable_llm_eval:
        llm_evaluator = LLMEvaluator(config, "", "")
        print(f"🔍 Using LLM evaluation with model: {config.evaluation.eval_model}")
        logger.info(f"🔍 Using LLM evaluation with model: {config.evaluation.eval_model}")
    else:
        evaluator_obj = Evaluator("", "", None)
        print("🔍 Using exact match evaluation")
        logger.info("🔍 Using exact match evaluation")
    
    # Statistics
    total_correct_llm = 0
    total_correct_exact = 0
    total_evaluated = 0

    def _strip_embeddings(items):
        if not items or not isinstance(items, list):
            return items
        cleaned = []
        for item in items:
            if isinstance(item, dict):
                cleaned.append({k: v for k, v in item.items() if k not in ("embedding", "embeddings")})
            else:
                cleaned.append(item)
        return cleaned

    
    for i in range(dataset_len):
        query = query_dataset[i]
        appended = False
        try:
            question_time = query.get("question_time", "")
            question_type = query.get("question_type", "")
            res = await graph_rag.query(
                query["question"],
                question_time=question_time,
                question_type=question_type,
            )
            
            # Extract answer and top_session_ids from result dictionary
            if isinstance(res, dict):
                query["output"] = res.get('answer', '')
                query["top_session_ids"] = res.get('top_session_ids', [])
                query["retrieval_calls"] = res.get('retrieval_calls', 0)
                query["retrieved_sessions"] = res.get('retrieved_sessions', len(query["top_session_ids"]))
                query["react_trace"] = res.get("react_trace", [])
                query["react_llm_calls_total"] = res.get("react_llm_calls_total", "")

                # Trace fields
                query["entities"] = res.get("entities", [])
                query["relevant_entities"] = res.get("relevant_entities", [])
                query["triples"] = _strip_embeddings(res.get("triples", []))
                query["chunks"] = res.get("chunks", [])
                query["summaries"] = _strip_embeddings(res.get("summaries", []))
                query["formatted_prompt"] = res.get("formatted_prompt", "")
                # ReAct multihop debug fields for context-budget auditing.
                query["react_round_chunk_counts"] = res.get("react_round_chunk_counts", [])
                query["final_context_chunk_count"] = res.get("final_context_chunk_count", "")
                query["final_context_chunk_alloc_per_turn"] = res.get(
                    "final_context_chunk_alloc_per_turn", []
                )
            else:
                query["output"] = res
                query["top_session_ids"] = []
                query["retrieval_calls"] = 0
                query["retrieved_sessions"] = 0
                query["react_trace"] = []
                query["react_llm_calls_total"] = ""
                query["react_round_chunk_counts"] = []
                query["final_context_chunk_count"] = ""
                query["final_context_chunk_alloc_per_turn"] = []

            # Model/metadata fields for trace/export
            query["model"] = (config.query_llm.model or config.llm.model) if config else ""
            query["memory_system"] = "LiCoMemory"
            query["trial"] = int(os.environ.get("TRIAL", "1"))
            query["llm_judge"] = None
            query["success"] = False
            memos_metrics = derive_memos_metrics(row=query, record=res if isinstance(res, dict) else query)
            query["context_tokens"] = int(memos_metrics["context_tokens"])
            query["response_duration_ms"] = float(memos_metrics["response_duration_ms"])
            query["search_duration_ms"] = float(memos_metrics["search_duration_ms"])
            query["total_duration_ms"] = float(memos_metrics["total_duration_ms"])
            query["memos_stats"] = {
                "context_tokens": int(memos_metrics["context_tokens"]),
                "response_duration_ms": float(memos_metrics["response_duration_ms"]),
                "search_duration_ms": float(memos_metrics["search_duration_ms"]),
                "total_duration_ms": float(memos_metrics["total_duration_ms"]),
            }
            if isinstance(res, dict) and "total_cost_usd" in res:
                query["total_cost_usd"] = res.get("total_cost_usd")

            all_res.append(query)
            appended = True
            
            # Real-time evaluation
            expected_answer = str(query.get('answer', '')).strip()
            model_output = str(query.get('output', '')).strip()
            
            if expected_answer and model_output:
                if _is_query_error_output(model_output):
                    separator = "=" * 80
                    skip_msg = f"Question {i+1}/{dataset_len}: ⚠️  Query failed before evaluation"
                    print(f"\n{separator}")
                    print(skip_msg)
                    logger.warning(skip_msg)
                    logger.warning(f"Query error output: {model_output}")
                    query["llm_judge"] = False
                    query["success"] = False
                    continue

                total_evaluated += 1
                
                # LLM evaluation
                if enable_llm_eval:
                    question_type = query.get('question_type', 'default')
                    judge_bundle = await llm_evaluator.evaluate_with_llm_bundle(
                        question=query.get('question', ''),
                        answer=expected_answer,
                        response=model_output,
                        question_type=question_type,
                    )
                    is_correct_llm = bool(judge_bundle["majority_label"])
                    query["llm_judge"] = is_correct_llm
                    if len(judge_bundle["judgments"]) > 1:
                        query["llm_judgments"] = dict(judge_bundle["judgments"])
                    query["success"] = is_correct_llm
                    if is_correct_llm:
                        total_correct_llm += 1
                else:
                    is_correct_exact = evaluator_obj._check_answer_match(expected_answer, model_output) if evaluator_obj else False
                    query["llm_judge"] = None
                    query["success"] = bool(is_correct_exact)
                    if is_correct_exact:
                        total_correct_exact += 1
                
                # Print and log results
                separator = "=" * 80
                print(f"\n{separator}")
                logger.info(separator)
                
                question_info = f"Question {i+1}/{dataset_len}:"
                print(question_info)
                logger.info(question_info)
                
                memos_metrics = derive_memos_metrics(record=res if isinstance(res, dict) else {})
                print("MemOS-style Statistics:")
                logger.info("MemOS-style Statistics:")
                print(f"  Context Tokens: {memos_metrics['context_tokens']}")
                logger.info(f"  Context Tokens: {memos_metrics['context_tokens']}")
                print(f"  Response Duration: {memos_metrics['response_duration_ms']:.2f} ms")
                logger.info(f"  Response Duration: {memos_metrics['response_duration_ms']:.2f} ms")
                print(f"  Search Duration: {memos_metrics['search_duration_ms']:.2f} ms")
                logger.info(f"  Search Duration: {memos_metrics['search_duration_ms']:.2f} ms")
                print(f"  Total Duration: {memos_metrics['total_duration_ms']:.2f} ms")
                logger.info(f"  Total Duration: {memos_metrics['total_duration_ms']:.2f} ms")
                
                # Print answers
                print(f"Expected: {expected_answer}")
                logger.info(f"Expected: {expected_answer}")
                print(f"Got: {model_output}")
                logger.info(f"Got: {model_output}")
                
                # Print accuracy
                if total_evaluated > 0:
                    accuracy = total_correct_llm / total_evaluated * 100 if enable_llm_eval else 0.0
                    accuracy_text = f"Current Accuracy: {total_correct_llm}/{total_evaluated} ({accuracy:.1f}%)"
                    print(accuracy_text)
                    logger.info(accuracy_text)
            else:
                separator = "=" * 80
                skip_msg = f"Question {i+1}/{dataset_len}: ⚠️  Skipped (missing answer or output)"
                print(f"\n{separator}")
                print(skip_msg)
                logger.warning(skip_msg)
                
        except Exception as e:
            separator = "=" * 80
            error_msg = f"❌ Error processing query {i+1}: {e}"
            print(f"\n{separator}")
            print(error_msg)
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            query["output"] = "Error processing query"
            query["top_session_ids"] = []
            query["llm_judge"] = False
            query["success"] = False
            if not appended:
                all_res.append(query)
            fail_fast = os.getenv("LICOMEMORY_FAIL_FAST_QUERY_ERRORS", "0").lower() in {"1", "true", "yes", "on"}
            if fail_fast:
                raise
    
    # Final summary
    separator = "=" * 80
    print(f"\n{separator}")
    logger.info(separator)
    
    summary_title = "📊 FINAL EVALUATION SUMMARY"
    print(summary_title)
    logger.info(summary_title)
    
    print(separator)
    logger.info(separator)
    
    total_q = f"Total Questions: {dataset_len}"
    print(total_q)
    logger.info(total_q)
    
    evaluated = f"Evaluated: {total_evaluated}"
    print(evaluated)
    logger.info(evaluated)
    
    if enable_llm_eval:
        if total_evaluated > 0:
            final_accuracy = f"Final Accuracy: {total_correct_llm}/{total_evaluated} ({total_correct_llm/total_evaluated*100:.2f}%)"
        else:
            final_accuracy = "Final Accuracy: N/A"
        print(final_accuracy)
        logger.info(final_accuracy)
    
    print(f"{separator}\n")
    logger.info(separator)
    
    return all_res

def wrapper_query(query_dataset, graph_rag, result_dir, config=None):
    """Process queries and save results."""
    dataset_len = len(query_dataset)
    print(f"Processing {dataset_len} queries...")
    
    # Use single event loop for all queries
    all_res = asyncio.run(process_queries_async(query_dataset, graph_rag, dataset_len, config))

    # Save results
    save_path = os.path.join(result_dir, "results.json")
    sanitized = []
    for record in all_res:
        memos_ready = sanitize_trace_record_for_export(record)
        sanitized.append(_sanitize_for_json(memos_ready))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {save_path}")

    return save_path

def redact_answer_field_in_results(path: str) -> None:
    """Remove ground-truth `answer` field from exported results after evaluation."""
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return
        changed = False
        for row in rows:
            if isinstance(row, dict) and "answer" in row:
                row.pop("answer", None)
                changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            logger.info("Redacted `answer` field from exported results: %s", path)
    except Exception as e:
            logger.warning("Failed to redact `answer` field for %s: %s", path, e)


def _aggregate_memos_summary_from_results(path: str) -> dict:
    if not path or not os.path.exists(path):
        return aggregate_memos_metrics([])
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return aggregate_memos_metrics([])
        return aggregate_memos_metrics(rows)
    except Exception as e:
        logger.warning("Failed to aggregate MemOS-style stats from %s: %s", path, e)
        return aggregate_memos_metrics([])


def _aggregate_query_report_from_results(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {
            "memos": aggregate_memos_metrics([]),
            "total_cost_usd": 0.0,
            "total_duration_sec": 0.0,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            rows = []
        memos = aggregate_memos_metrics(rows)
        total_cost_usd = 0.0
        total_duration_sec = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                total_cost_usd += float(row.get("total_cost_usd") or 0.0)
            except Exception:
                pass
            try:
                total_duration_sec += float(row.get("total_duration_ms") or 0.0) / 1000.0
            except Exception:
                pass
        return {
            "memos": memos,
            "total_cost_usd": round(total_cost_usd, 4),
            "total_duration_sec": round(total_duration_sec, 3),
        }
    except Exception as e:
        logger.warning("Failed to aggregate query report from %s: %s", path, e)
        return {
            "memos": aggregate_memos_metrics([]),
            "total_cost_usd": 0.0,
            "total_duration_sec": 0.0,
        }

async def wrapper_evaluation(path, dataset_name, result_dir, config=None):
    """Run evaluation on results."""
    _load_runtime_dependencies()
    eval = Evaluator(path, dataset_name, config)

    # Query stage already performs per-sample LLM judge when enabled.
    # Force post-eval to consume cached labels to avoid a second judge API pass.
    if config and config.evaluation.enable_llm_eval:
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
            cached_metrics = eval._calculate_metrics_from_cached_llm_judge(results)
            if cached_metrics is None:
                logger.info(
                    "Cached llm_judge labels are incomplete; "
                    "running the configured LLM judge path instead of using any fallback metrics."
                )
                answer_metrics = await eval.evaluate()
            else:
                answer_metrics = cached_metrics
            matching_metrics = eval.session_matching_evaluator.evaluate_all(results)
            res_dict = {
                **answer_metrics,
                "average_matching_score": matching_metrics.get("average_matching_score", 0.0),
                "matching_scores": matching_metrics.get("matching_scores", []),
            }
        except Exception as e:
            logger.warning(
                "Cached-label post-eval path failed (%s); rerunning the configured evaluator path.", e
            )
            res_dict = await eval.evaluate()
    else:
        res_dict = await eval.evaluate()

    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(res_dict, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to {save_path}")
    return res_dict

if __name__ == "__main__":
    args = parse_args()
    _load_runtime_dependencies()

    # Load configuration
    config = Config.parse(Path(args.opt), dataset_name=args.dataset_name)

    # Create directories first
    base_dir, result_dir = check_dirs(config, args.root)
    
    # Update logger path to use the new base directory
    update_logger_path(base_dir)
    
    # Initialize GraphRAG with base directory
    graph_rag = GraphRAG(config, base_dir)

    # Load dataset
    try:
        query_dataset = RAGQueryDataset(
            data_dir=os.path.join(config.data_root, config.dataset_name)
        )
        corpus = query_dataset.get_corpus()
        print(f"Loaded dataset with {len(corpus)} documents")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        corpus = []

    # Check if we should insert documents or load existing graph
    force = getattr(config.graph, 'force', False)
    add = getattr(config.graph, 'add', False)
    
    if not force and not add:
        print("Load existing mode: skipping document insertion, will load from pkl file")
        try:
            attach_corpus_chunks = bool(
                getattr(config.retriever, "load_only_attach_corpus_chunks", False)
            )
            insert_corpus = corpus if attach_corpus_chunks else []
            asyncio.run(graph_rag.insert(insert_corpus))
        except Exception as e:
            print(f"Load existing graph failed: {e}")
            allow_rebuild = os.getenv("LICOMEMORY_REBUILD_ON_CORRUPT_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}
            if allow_rebuild and corpus:
                print("Falling back to rebuild graph from corpus for this sample...")
                # Mutate runtime config so insert() takes the build path.
                config.graph.force = True
                config.graph.add = False
                asyncio.run(graph_rag.insert(corpus))
                print("Fallback rebuild completed")
            else:
                print("Set LICOMEMORY_REBUILD_ON_CORRUPT_CACHE=1 to allow automatic rebuild fallback.")
                raise
    elif corpus:
        print("Inserting documents into graph...")
        asyncio.run(graph_rag.insert(corpus))
        print("Document insertion completed")
    else:
        print("No corpus provided and not in load existing mode")

    # Initialize final report generator
    final_report = FinalReportGenerator()
    
    # Collect graph building statistics
    if hasattr(graph_rag.core, 'dynamic_memory') and graph_rag.core.dynamic_memory:
        dm = graph_rag.core.dynamic_memory
        if hasattr(dm, 'time_manager') and hasattr(dm, 'cost_manager'):
            final_report.set_graph_building_stats(
                dm.time_manager.get_graph_building_summary(),
                dm.cost_manager.get_graph_building_summary()
            )

    # Run queries if requested
    run_query = str(args.query).strip().lower() in ("1", "true", "yes", "y") if args.query is not None else False
    if run_query:
        print("Running query and evaluation...")
        save_path = wrapper_query(query_dataset, graph_rag, result_dir, config)
        evaluation_results = asyncio.run(wrapper_evaluation(save_path, config.dataset_name, result_dir, config))
        # Keep evaluation correctness while removing ground-truth answers from exported artifacts.
        redact_answer_field_in_results(save_path)
        final_report.set_evaluation_results(evaluation_results)
        
        # Collect query statistics from query processor
        query_report = _aggregate_query_report_from_results(save_path)
        final_report.add_runtime_stats(
            {"total_query_time": query_report["total_duration_sec"]},
            {"total_cost_usd": query_report["total_cost_usd"]},
            query_report["memos"],
        )
    else:
        print("Skipping query and evaluation.")

    try:
        final_report_path = os.path.join(result_dir, "final_report.json")
        final_report_payload = final_report.generate_comprehensive_report()
        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(final_report_payload, f, ensure_ascii=False, indent=2)
        logger.info("Final report saved to %s", final_report_path)
    except Exception as e:
        logger.warning("Failed to save final_report.json for %s: %s", config.dataset_name, e)

    print("Dynamic Memory Graph RAG System execution completed!")
