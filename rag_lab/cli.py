"""Command-line interface for RAG experiments."""

import argparse
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import tyro
from tqdm import tqdm

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


def create_run_config(config: Config, retriever_type: str, k: int, max_samples: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = None) -> Dict:
    """Create a minimal config containing only run-specific information."""
    run_config = {
        "experiment": {
            "retriever_type": retriever_type,
            "k": k,
            "max_samples": max_samples,
            "shuffle": shuffle,
            "shuffle_seed": seed if seed is not None else config.seed,
            "config_seed": config.seed,
            "dataset_name": config.dataset.name,
            "dataset_split": config.dataset.split
        },
        "generator": {
            "model_name": config.generator.model_name,
            "max_new_tokens": config.generator.max_new_tokens,
            "temperature": config.generator.temperature,
            "top_p": config.generator.top_p,
            "context_token_budget": config.generator.context_token_budget,
            "device": config.generator.device.device,
            "torch_dtype": config.generator.device.torch_dtype
        }
    }
    
    # Add retriever-specific config only for the retriever being used
    if retriever_type != "none":
        retriever_config = config.get_retriever_config(retriever_type)
        
        if retriever_type == "boolean":
            run_config["retriever"] = {
                "type": "boolean",
                "normalize_tokens": retriever_config.normalize_tokens,
                "case_sensitive": retriever_config.case_sensitive
            }
        elif retriever_type == "tfidf":
            run_config["retriever"] = {
                "type": "tfidf",
                "ngram_range": retriever_config.ngram_range,
                "min_df": retriever_config.min_df,
                "max_df": retriever_config.max_df,
                "norm": retriever_config.norm
            }
        elif retriever_type == "bm25":
            run_config["retriever"] = {
                "type": "bm25",
                "k1": retriever_config.k1,
                "b": retriever_config.b
            }
        elif retriever_type == "dense":
            run_config["retriever"] = {
                "type": "dense",
                "model_name": retriever_config.model_name,
                "normalize_embeddings": retriever_config.normalize_embeddings,
                "faiss_index_type": retriever_config.faiss_index_type,
                "batch_size": retriever_config.batch_size
            }
        elif retriever_type == "sota":
            run_config["retriever"] = {
                "type": "sota",
                "model_name": retriever_config.model_name,
                "normalize_embeddings": retriever_config.normalize_embeddings,
                "faiss_index_type": retriever_config.faiss_index_type,
                "batch_size": retriever_config.batch_size,
                "k_rerank": retriever_config.k_rerank,
                "cross_encoder_model": retriever_config.cross_encoder_model,
                "rerank_batch_size": retriever_config.rerank_batch_size
            }
    else:
        run_config["retriever"] = {
            "type": "none",
            "description": "Direct generation without retrieval"
        }
    
    return run_config


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
            case_sensitive=retriever_config.case_sensitive,
            min_term_threshold=retriever_config.min_term_threshold
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


def build_index(config: Config, dataset: str, retriever_type: Optional[str] = None):
    """Build index for a specific retriever and dataset, or all retrievers if none specified."""
    
    # Define all available retrievers (excluding 'none' since it doesn't need an index)
    all_retrievers = ["boolean", "tfidf", "bm25", "dense", "sota"]
    
    if retriever_type is None:
        # Build all retrievers
        print(f"Building indexes for all retrievers on {dataset} dataset...")
        print(f"Retrievers to build: {', '.join(all_retrievers)}")
        
        # Load dataset once for all retrievers
        dataset_loader = SquadDataset(config.dataset)
        dataset_loader.load()
        corpus = dataset_loader.get_corpus()
        print(f"Corpus size: {len(corpus)} documents")
        
        for i, ret_type in enumerate(all_retrievers, 1):
            print(f"\n{'='*70}")
            print(f"🔧 BUILDING INDEX {i}/{len(all_retrievers)}: {ret_type.upper()}")
            print(f"📊 Progress: {i/len(all_retrievers)*100:.1f}% complete")
            print(f"{'='*70}")
            
            try:
                _build_single_index(config, dataset, ret_type, corpus)
                print(f"✅ COMPLETED: {ret_type} index built successfully")
            except Exception as e:
                print(f"❌ ERROR: Failed to build {ret_type} index: {e}")
                continue
        
        print(f"\n{'='*70}")
        print("🎉 INDEX BUILDING COMPLETE")
        print(f"{'='*70}")
        
    else:
        # Build single retriever
        print(f"Building index for {retriever_type} retriever on {dataset} dataset...")
        
        # Load dataset
        dataset_loader = SquadDataset(config.dataset)
        dataset_loader.load()
        corpus = dataset_loader.get_corpus()
        print(f"Corpus size: {len(corpus)} documents")
        
        _build_single_index(config, dataset, retriever_type, corpus)


def _build_single_index(config: Config, dataset: str, retriever_type: str, corpus: List[str]):
    """Build index for a single retriever."""
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
                  max_samples: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = None):
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
        # Auto-enable shuffle if seed is provided
        if seed is not None:
            shuffle = True
            
        if shuffle:
            # Use provided seed or fall back to config seed
            shuffle_seed = seed if seed is not None else config.seed
            random.seed(shuffle_seed)
            qa_pairs_copy = qa_pairs.copy()
            random.shuffle(qa_pairs_copy)
            qa_pairs = qa_pairs_copy[:max_samples]
            print(f"Shuffled and sampled {len(qa_pairs)} QA pairs (seed: {shuffle_seed})")
        else:
            qa_pairs = qa_pairs[:max_samples]
            print(f"Sampled first {len(qa_pairs)} QA pairs")
    
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
    
    # Create progress bar
    progress_bar = tqdm(qa_pairs, desc=f"🔍 {retriever_type.upper()} Retrieval + 🤖 Generation", 
                       unit="QA", ncols=120, 
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    
    for i, qa_pair in enumerate(progress_bar):
        
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
        
        # Find ground truth document ID and text
        ground_truth_doc_id = dataset_loader.find_ground_truth_doc_id(qa_pair)
        ground_truth_corpus_id = dataset_loader.find_ground_truth_corpus_id(qa_pair)
        ground_truth_doc_text = None
        if ground_truth_doc_id is not None and ground_truth_doc_id < len(dataset_loader.documents):
            ground_truth_doc_text = dataset_loader.documents[ground_truth_doc_id].text
        
        # Store result
        result = {
            "qa_id": qa_pair.id,
            "question": qa_pair.question,
            "ground_truth": qa_pair.answer,
            "ground_truth_answers": qa_pair.answers,
            "ground_truth_doc_id": ground_truth_doc_id,
            "ground_truth_corpus_id": ground_truth_corpus_id,
            "ground_truth_doc_text": ground_truth_doc_text,
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
        
        # Update progress bar with current stats
        avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
        avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
        
        progress_bar.set_postfix({
            "Ret": f"{retrieval_times[-1]:.3f}s",
            "Gen": f"{generation_times[-1]:.3f}s", 
            "AvgRet": f"{avg_retrieval:.3f}s",
            "AvgGen": f"{avg_generation:.3f}s"
        })
        
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
        # Use actual ground truth documents as relevant documents (using corpus IDs)
        relevant_docs_list = []
        for result in results:
            if result["ground_truth_corpus_id"] is not None:
                relevant_docs_list.append({result["ground_truth_corpus_id"]})
            else:
                relevant_docs_list.append(set())  # No relevant docs if ground truth not found
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
    
    # Create minimal run-specific config
    run_config = create_run_config(config, retriever_type, k, max_samples, shuffle, seed)
    
    save_experiment_results(
        {
            "metrics": all_metrics,
            "predictions": results,
            "config": run_config
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


def run_experiment_with_qa_pairs(config: Config, dataset: str, retriever_type: str, k: int,
                                dataset_loader: SquadDataset, qa_pairs: List, max_samples: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = None) -> Tuple[Dict, List]:
    """Run a single RAG experiment with pre-selected QA pairs."""
    print(f"Running experiment: {retriever_type} retriever, k={k}")
    print(f"Evaluating on {len(qa_pairs)} QA pairs (pre-selected for consistency)")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get corpus and retriever (dataset already loaded)
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
    
    # Create progress bar
    progress_bar = tqdm(qa_pairs, desc=f"🔍 {retriever_type.upper()} Retrieval + 🤖 Generation", 
                       unit="QA", ncols=120, 
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    
    for i, qa_pair in enumerate(progress_bar):
        
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
        
        # Find ground truth document ID and text
        ground_truth_doc_id = dataset_loader.find_ground_truth_doc_id(qa_pair)
        ground_truth_corpus_id = dataset_loader.find_ground_truth_corpus_id(qa_pair)
        ground_truth_doc_text = None
        if ground_truth_doc_id is not None and ground_truth_doc_id < len(dataset_loader.documents):
            ground_truth_doc_text = dataset_loader.documents[ground_truth_doc_id].text
        
        # Store result
        result = {
            "qa_id": qa_pair.id,
            "question": qa_pair.question,
            "ground_truth": qa_pair.answer,
            "ground_truth_answers": qa_pair.answers,
            "ground_truth_doc_id": ground_truth_doc_id,
            "ground_truth_corpus_id": ground_truth_corpus_id,
            "ground_truth_doc_text": ground_truth_doc_text,
            "answer": generation_result["answer"],
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
        
        # Update progress bar with current stats
        avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
        avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
        
        progress_bar.set_postfix({
            "Ret": f"{retrieval_times[-1]:.3f}s",
            "Gen": f"{generation_times[-1]:.3f}s", 
            "AvgRet": f"{avg_retrieval:.3f}s",
            "AvgGen": f"{avg_generation:.3f}s"
        })
        
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
        # Use actual ground truth documents as relevant documents (using corpus IDs)
        relevant_docs_list = []
        for result in results:
            if result["ground_truth_corpus_id"] is not None:
                relevant_docs_list.append({result["ground_truth_corpus_id"]})
            else:
                relevant_docs_list.append(set())  # No relevant docs if ground truth not found
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
    
    # Save experiment results with minimal config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{retriever_type}_k{k}"
    
    # Create minimal run-specific config
    run_config = create_run_config(config, retriever_type, k, max_samples, shuffle, seed)
    
    save_experiment_results(
        {
            "metrics": all_metrics,
            "predictions": results,
            "config": run_config
        },
        config.output.base_dir,
        experiment_name,
        timestamp
    )
    
    # Print results
    print("="*50)
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
              dataset: str = "squad", max_samples: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = None):
    """Run experiments across multiple retrievers and k values."""
    print(f"Running sweep: {retrievers} retrievers, k={k_values}")
    
    # Load dataset once and select consistent samples for all experiments
    print("Loading dataset and selecting consistent samples for all experiments...")
    set_seed(config.seed)  # Ensure deterministic sample selection
    
    dataset_loader = SquadDataset(config.dataset)
    dataset_loader.load()
    
    # Get the same QA pairs that will be used for all experiments
    all_qa_pairs = dataset_loader.get_qa_pairs()
    if max_samples:
        # Auto-enable shuffle if seed is provided
        if seed is not None:
            shuffle = True
            
        if shuffle:
            # Use provided seed or fall back to config seed
            shuffle_seed = seed if seed is not None else config.seed
            random.seed(shuffle_seed)
            qa_pairs_copy = all_qa_pairs.copy()
            random.shuffle(qa_pairs_copy)
            selected_qa_pairs = qa_pairs_copy[:max_samples]
            print(f"Shuffled and sampled {len(selected_qa_pairs)} QA pairs (seed: {shuffle_seed})")
        else:
            selected_qa_pairs = all_qa_pairs[:max_samples]
            print(f"Sampled first {len(selected_qa_pairs)} QA pairs")
    else:
        selected_qa_pairs = all_qa_pairs
    
    print(f"Selected {len(selected_qa_pairs)} QA pairs for consistent evaluation across all retrievers")
    
    # Set up results storage
    sweep_results = []
    
    # Calculate total experiments for progress tracking
    total_experiments = len(retrievers) * len(k_values)
    experiment_count = 0
    
    # Run experiments with the same QA pairs
    for retriever_type in retrievers:
        for k in k_values:
            experiment_count += 1
            print(f"\n{'='*70}")
            print(f"🧪 EXPERIMENT {experiment_count}/{total_experiments}: {retriever_type.upper()} retriever, k={k}")
            print(f"📊 Progress: {experiment_count/total_experiments*100:.1f}% complete")
            print(f"{'='*70}")
            
            try:
                # Run experiment with pre-selected QA pairs
                metrics, _ = run_experiment_with_qa_pairs(config, dataset, retriever_type, k, 
                                                        dataset_loader, selected_qa_pairs, max_samples, shuffle, seed)
                
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
                
                # Show completion status
                print(f"✅ COMPLETED: {retriever_type} k={k} | EM: {metrics['exact_match']:.3f} | F1: {metrics['f1']:.3f} | Ret: {metrics['avg_retrieval_time']:.3f}s | Gen: {metrics['avg_generation_time']:.3f}s")
                
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
    build_parser.add_argument("--retriever", 
                             choices=["boolean", "tfidf", "bm25", "dense", "sota"],
                             help="Retriever type (builds all if not specified)")
    
    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run single experiment")
    run_parser.add_argument("--dataset", default="squad", help="Dataset name")
    run_parser.add_argument("--retriever", required=True,
                           choices=["none", "boolean", "tfidf", "bm25", "dense", "sota"],
                           help="Retriever type")
    run_parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    run_parser.add_argument("--max-samples", type=int, help="Maximum number of QA pairs to evaluate")
    run_parser.add_argument("--shuffle", action="store_true", help="Shuffle data before sampling")
    run_parser.add_argument("--seed", type=int, help="Random seed for shuffling (overrides config seed)")
    
    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run experiment sweep")
    sweep_parser.add_argument("--retrievers", nargs="+", 
                             choices=["none", "boolean", "tfidf", "bm25", "dense", "sota"],
                             default=["none", "boolean", "tfidf", "bm25", "dense", "sota"],
                             help="Retriever types to test")
    sweep_parser.add_argument("--k", nargs="+", type=int, default=[3],
                             help="k values to test")
    sweep_parser.add_argument("--dataset", default="squad", help="Dataset name")
    sweep_parser.add_argument("--max-samples", type=int, help="Maximum number of QA pairs to evaluate")
    sweep_parser.add_argument("--shuffle", action="store_true", help="Shuffle data before sampling")
    sweep_parser.add_argument("--seed", type=int, help="Random seed for shuffling (overrides config seed)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = Config.from_yaml()
    config.setup_directories()
    
    # Execute command
    if args.command == "build-index":
        build_index(config, args.dataset, args.retriever)
    
    elif args.command == "run":
        run_experiment(config, args.dataset, args.retriever, args.k, args.max_samples, args.shuffle, args.seed)
    
    elif args.command == "sweep":
        run_sweep(config, args.retrievers, args.k, args.dataset, args.max_samples, args.shuffle, args.seed)


if __name__ == "__main__":
    main()
