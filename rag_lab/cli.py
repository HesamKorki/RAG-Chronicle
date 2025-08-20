"""Command-line interface for RAG experiments."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tyro

from .config import Config
from .data import SquadDataset
from .generator.qwen import QwenGenerator
from .retrievers.boolean import BooleanRetriever
from .retrievers.tfidf import TfidfRetriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever
from .retrievers.sota import SOTARetriever
from .eval.squad_metrics import evaluate_squad_batch
from .eval.retrieval_metrics import evaluate_retrieval_batch, compute_retrieval_latency
from .utils.seeds import set_seed
from .utils.timing import Timer, PerformanceTracker
from .utils.io import save_experiment_results, save_csv


def get_retriever(retriever_type: str, config: Config):
    """Get retriever instance based on type."""
    retriever_map = {
        "boolean": BooleanRetriever,
        "tfidf": TfidfRetriever,
        "bm25": BM25Retriever,
        "dense": DenseRetriever,
        "sota": SOTARetriever
    }
    
    if retriever_type not in retriever_map:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    retriever_class = retriever_map[retriever_type]
    retriever_config = config.get_retriever_config(retriever_type)
    

    
    # Initialize with config parameters
    if retriever_type == "boolean":
        return retriever_class(
            normalize_tokens=retriever_config.normalize_tokens,
            case_sensitive=retriever_config.case_sensitive
        )
    elif retriever_type == "tfidf":
        return retriever_class(
            ngram_range=retriever_config.ngram_range,
            min_df=retriever_config.min_df,
            max_df=retriever_config.max_df,
            norm=retriever_config.norm
        )
    elif retriever_type == "bm25":
        return retriever_class(
            k1=retriever_config.k1,
            b=retriever_config.b
        )
    elif retriever_type == "dense":
        return retriever_class(
            model_name=retriever_config.model_name,
            normalize_embeddings=retriever_config.normalize_embeddings,
            faiss_index_type=retriever_config.faiss_index_type,
            batch_size=retriever_config.batch_size
        )
    elif retriever_type == "sota":
        return retriever_class(
            model_name=retriever_config.model_name,
            normalize_embeddings=retriever_config.normalize_embeddings,
            faiss_index_type=retriever_config.faiss_index_type,
            batch_size=retriever_config.batch_size,
            k_rerank=retriever_config.k_rerank,
            cross_encoder_model=retriever_config.cross_encoder_model,
            rerank_batch_size=retriever_config.rerank_batch_size
        )
    
    return retriever_class()


def build_index(config: Config, dataset: str, retriever_type: str):
    """Build index for a specific retriever and dataset."""
    print(f"Building index for {retriever_type} retriever on {dataset} dataset...")
    
    # Load dataset
    dataset_loader = SquadDataset(config.dataset)
    dataset_loader.load()
    
    # Get corpus
    corpus = dataset_loader.get_corpus()
    print(f"Corpus size: {len(corpus)} documents")
    
    # Get retriever
    retriever = get_retriever(retriever_type, config)
    
    # Build index
    with Timer("Indexing"):
        retriever.index(corpus)
    
    # Save index
    index_path = config.output.get_index_dir(dataset, retriever_type)
    retriever.save_index(index_path)
    
    # Print index stats
    stats = retriever.get_index_stats()
    print(f"Index statistics: {stats}")


def run_experiment(config: Config, dataset: str, retriever_type: str, k: int = 5, 
                  max_samples: Optional[int] = None):
    """Run a single RAG experiment."""
    print(f"Running experiment: {retriever_type} retriever, k={k}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Load dataset
    dataset_loader = SquadDataset(config.dataset)
    dataset_loader.load()
    
    # Get QA pairs
    qa_pairs = dataset_loader.get_qa_pairs()
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    
    print(f"Evaluating on {len(qa_pairs)} QA pairs")
    
    # Get corpus and retriever
    corpus = dataset_loader.get_corpus()
    
    if retriever_type == "none":
        print("Running in direct generation mode (no retrieval)")
        retriever = None
    else:
        # Get retriever
        retriever = get_retriever(retriever_type, config)
        
        # Load or build index
        index_path = config.output.get_index_dir(dataset, retriever_type)
        if index_path.exists():
            print(f"Loading existing index from {index_path}")
            retriever.load_index(index_path)
        else:
            print(f"Building new index at {index_path}")
            retriever.index(corpus)
            retriever.save_index(index_path)
    
    # Initialize generator
    generator = QwenGenerator(config.generator)
    
    # Performance tracking
    perf_tracker = PerformanceTracker()
    
    # Run experiments
    results = []
    retrieval_times = []
    generation_times = []
    
    for i, qa_pair in enumerate(qa_pairs):
        if i % 100 == 0:
            print(f"Processing QA pair {i+1}/{len(qa_pairs)}")
        
        # Retrieve documents (if retriever exists)
        if retriever is not None:
            with Timer("Retrieval") as timer:
                retrieved_docs = retriever.retrieve(qa_pair.question, k)
            retrieval_times.append(timer.elapsed_time)
            
            # Get passages
            passages = []
            doc_ids = []
            for doc_id, score in retrieved_docs:
                if doc_id < len(corpus):
                    passages.append(corpus[doc_id])
                    doc_ids.append(doc_id)
        else:
            # No retrieval - direct generation
            retrieval_times.append(0.0)  # No retrieval time
            passages = []
            doc_ids = []
            retrieved_docs = []
        
        # Generate answer
        with Timer("Generation") as timer:
            generation_result = generator.generate(qa_pair.question, passages, doc_ids, k)
        generation_times.append(timer.elapsed_time)
        
        # Store result
        result = {
            "qa_id": qa_pair.id,
            "question": qa_pair.question,
            "ground_truth": qa_pair.answer,
            "ground_truth_answers": qa_pair.answers,
            "answer": generation_result["answer"],  # Changed from predicted_answer to match evaluation expectation
            "predicted_answer": generation_result["answer"],
            "retrieved_docs": doc_ids,
            "retrieval_scores": [score for _, score in retrieved_docs],
            "passages_used": generation_result["passages_used"],
            "retrieval_time": retrieval_times[-1],
            "generation_time": generation_times[-1],
            "tokens_generated": generation_result["tokens_generated"],
            "is_impossible": qa_pair.is_impossible
        }
        
        results.append(result)
        
        # Track performance
        perf_tracker.add_metric("retrieval_time", retrieval_times[-1])
        perf_tracker.add_metric("generation_time", generation_times[-1])
    
    # Evaluate results
    print("Evaluating results...")
    
    # SQuAD metrics
    squad_metrics = evaluate_squad_batch(results, qa_pairs)
    
    # Retrieval metrics (only if retriever was used)
    if retriever is not None:
        retrieved_docs_list = [result["retrieved_docs"] for result in results]
        # For now, assume all documents are equally relevant (simplified)
        relevant_docs_list = [set(range(len(corpus))) for _ in results]
        retrieval_metrics = evaluate_retrieval_batch(retrieved_docs_list, relevant_docs_list, [k])
    else:
        # No retrieval metrics for direct generation
        retrieval_metrics = {}
    
    # Performance metrics
    performance_metrics = {
        "avg_retrieval_time": perf_tracker.get_stats("retrieval_time")["mean"],
        "avg_generation_time": perf_tracker.get_stats("generation_time")["mean"],
        "total_time": sum(retrieval_times) + sum(generation_times)
    }
    
    # Combine all metrics
    all_metrics = {
        **squad_metrics,
        **retrieval_metrics,
        **performance_metrics
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{retriever_type}_k{k}"
    
    save_experiment_results(
        {
            "metrics": all_metrics,
            "predictions": results,
            "config": config.dict()
        },
        config.output.base_dir,
        experiment_name,
        timestamp
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Retriever: {retriever_type}")
    print(f"k: {k}")
    print(f"Dataset: {dataset}")
    print(f"QA pairs: {len(qa_pairs)}")
    print(f"Exact Match: {squad_metrics['exact_match']:.4f}")
    print(f"F1 Score: {squad_metrics['f1']:.4f}")
    print(f"Recall@{k}: {retrieval_metrics.get(f'recall@{k}', 0):.4f}")
    print(f"MRR@{k}: {retrieval_metrics.get(f'mrr@{k}', 0):.4f}")
    print(f"Avg Retrieval Time: {performance_metrics['avg_retrieval_time']:.4f}s")
    print(f"Avg Generation Time: {performance_metrics['avg_generation_time']:.4f}s")
    print("="*50)
    
    return all_metrics, results


def run_sweep(config: Config, retrievers: List[str], k_values: List[int], 
              dataset: str = "squad", max_samples: Optional[int] = None):
    """Run experiments across multiple retrievers and k values."""
    print(f"Running sweep: {retrievers} retrievers, k={k_values}")
    
    # Set up results storage
    sweep_results = []
    
    # Run experiments
    for retriever_type in retrievers:
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Running: {retriever_type} retriever, k={k}")
            print(f"{'='*60}")
            
            try:
                metrics, _ = run_experiment(config, dataset, retriever_type, k, max_samples)
                
                # Store result
                sweep_results.append({
                    "retriever": retriever_type,
                    "k": k,
                    "exact_match": metrics["exact_match"],
                    "f1": metrics["f1"],
                    f"recall@{k}": metrics.get(f"recall@{k}", 0),
                    f"mrr@{k}": metrics.get(f"mrr@{k}", 0),
                    f"ndcg@{k}": metrics.get(f"ndcg@{k}", 0),
                    "avg_retrieval_time": metrics["avg_retrieval_time"],
                    "avg_generation_time": metrics["avg_generation_time"]
                })
                
            except Exception as e:
                print(f"Error running {retriever_type} with k={k}: {e}")
                continue
    
    # Save sweep results
    if sweep_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_path = config.output.base_dir / timestamp / "sweep_results.csv"
        save_csv(sweep_results, sweep_path)
        
        # Print summary table
        print("\n" + "="*80)
        print("SWEEP RESULTS SUMMARY")
        print("="*80)
        print(f"{'Retriever':<12} {'k':<3} {'EM':<8} {'F1':<8} {'R@k':<8} {'MRR@k':<8} {'NDCG@k':<8}")
        print("-"*80)
        
        for result in sweep_results:
            k = result["k"]
            print(f"{result['retriever']:<12} {k:<3} {result['exact_match']:<8.4f} "
                  f"{result['f1']:<8.4f} {result[f'recall@{k}']:<8.4f} "
                  f"{result[f'mrr@{k}']:<8.4f} {result[f'ndcg@{k}']:<8.4f}")
        
        print("="*80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Lab - RAG Experiment Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build index command
    build_parser = subparsers.add_parser("build-index", help="Build retriever index")
    build_parser.add_argument("--dataset", default="squad", help="Dataset name")
    build_parser.add_argument("--retriever", required=True, 
                             choices=["boolean", "tfidf", "bm25", "dense", "sota"],
                             help="Retriever type")
    
    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run single experiment")
    run_parser.add_argument("--dataset", default="squad", help="Dataset name")
    run_parser.add_argument("--retriever", required=True,
                           choices=["none", "boolean", "tfidf", "bm25", "dense", "sota"],
                           help="Retriever type")
    run_parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    run_parser.add_argument("--max-samples", type=int, help="Maximum number of QA pairs to evaluate")
    
    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run experiment sweep")
    sweep_parser.add_argument("--retrievers", nargs="+", 
                             choices=["none", "boolean", "tfidf", "bm25", "dense", "sota"],
                             default=["boolean", "tfidf", "bm25", "dense", "sota"],
                             help="Retriever types to test")
    sweep_parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                             help="k values to test")
    sweep_parser.add_argument("--dataset", default="squad", help="Dataset name")
    sweep_parser.add_argument("--max-samples", type=int, help="Maximum number of QA pairs to evaluate")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = Config()
    config.setup_directories()
    
    # Execute command
    if args.command == "build-index":
        build_index(config, args.dataset, args.retriever)
    
    elif args.command == "run":
        run_experiment(config, args.dataset, args.retriever, args.k, args.max_samples)
    
    elif args.command == "sweep":
        run_sweep(config, args.retrievers, args.k, args.dataset, args.max_samples)


if __name__ == "__main__":
    main()
