"""Retrieval evaluation metrics for RAG experiments."""

import numpy as np
from typing import List, Dict, Any, Tuple, Set


def compute_recall_at_k(retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
    """Compute Recall@k."""
    if not relevant_docs:
        return 0.0
    
    retrieved_at_k = set(retrieved_docs[:k])
    relevant_retrieved = len(retrieved_at_k & relevant_docs)
    
    return relevant_retrieved / len(relevant_docs)


def compute_precision_at_k(retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
    """Compute Precision@k."""
    if k == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved_docs[:k])
    relevant_retrieved = len(retrieved_at_k & relevant_docs)
    
    return relevant_retrieved / k


def compute_mrr_at_k(retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
    """Compute Mean Reciprocal Rank@k."""
    if not relevant_docs:
        return 0.0
    
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    
    return 0.0


def compute_ndcg_at_k(retrieved_docs: List[int], relevant_docs: Set[int], k: int, 
                     relevance_scores: Dict[int, float] = None) -> float:
    """Compute nDCG@k."""
    if k == 0:
        return 0.0
    
    # Use binary relevance if no scores provided
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
    
    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        relevance = relevance_scores.get(doc_id, 0.0)
        dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Compute IDCG (ideal DCG)
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_hit_at_k(retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
    """Compute Hit@k (binary: 1 if any relevant doc is in top-k, 0 otherwise)."""
    retrieved_at_k = set(retrieved_docs[:k])
    return 1.0 if retrieved_at_k & relevant_docs else 0.0


def evaluate_retrieval_batch(retrieved_docs_list: List[List[int]], 
                           relevant_docs_list: List[Set[int]], 
                           k_values: List[int] = None) -> Dict[str, float]:
    """Evaluate retrieval performance for a batch of queries."""
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    if len(retrieved_docs_list) != len(relevant_docs_list):
        raise ValueError("Number of retrieved docs must match number of relevant docs")
    
    metrics = {}
    
    for k in k_values:
        recalls = []
        precisions = []
        mrrs = []
        ndcgs = []
        hits = []
        
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            recalls.append(compute_recall_at_k(retrieved_docs, relevant_docs, k))
            precisions.append(compute_precision_at_k(retrieved_docs, relevant_docs, k))
            mrrs.append(compute_mrr_at_k(retrieved_docs, relevant_docs, k))
            ndcgs.append(compute_ndcg_at_k(retrieved_docs, relevant_docs, k))
            hits.append(compute_hit_at_k(retrieved_docs, relevant_docs, k))
        
        metrics[f"recall@{k}"] = np.mean(recalls)
        metrics[f"precision@{k}"] = np.mean(precisions)
        metrics[f"mrr@{k}"] = np.mean(mrrs)
        metrics[f"ndcg@{k}"] = np.mean(ndcgs)
        metrics[f"hit@{k}"] = np.mean(hits)
        
        # Add standard deviations
        metrics[f"recall@{k}_std"] = np.std(recalls)
        metrics[f"precision@{k}_std"] = np.std(precisions)
        metrics[f"mrr@{k}_std"] = np.std(mrrs)
        metrics[f"ndcg@{k}_std"] = np.std(ndcgs)
        metrics[f"hit@{k}_std"] = np.std(hits)
    
    return metrics


def compute_retrieval_latency(retrieval_times: List[float]) -> Dict[str, float]:
    """Compute retrieval latency statistics."""
    if not retrieval_times:
        return {}
    
    return {
        "avg_retrieval_time": np.mean(retrieval_times),
        "median_retrieval_time": np.median(retrieval_times),
        "min_retrieval_time": np.min(retrieval_times),
        "max_retrieval_time": np.max(retrieval_times),
        "std_retrieval_time": np.std(retrieval_times),
        "p95_retrieval_time": np.percentile(retrieval_times, 95),
        "p99_retrieval_time": np.percentile(retrieval_times, 99)
    }


def compute_context_usage_stats(context_lengths: List[int], token_budgets: List[int]) -> Dict[str, float]:
    """Compute statistics about context usage."""
    if not context_lengths:
        return {}
    
    usage_ratios = [length / budget for length, budget in zip(context_lengths, token_budgets)]
    
    return {
        "avg_context_length": np.mean(context_lengths),
        "avg_context_usage_ratio": np.mean(usage_ratios),
        "median_context_usage_ratio": np.median(usage_ratios),
        "max_context_usage_ratio": np.max(usage_ratios),
        "context_usage_std": np.std(usage_ratios)
    }


def analyze_retrieval_diversity(retrieved_docs_list: List[List[int]], 
                              corpus_size: int) -> Dict[str, float]:
    """Analyze diversity of retrieved documents."""
    if not retrieved_docs_list:
        return {}
    
    # Count document frequencies
    doc_frequencies = {}
    total_retrievals = 0
    
    for retrieved_docs in retrieved_docs_list:
        for doc_id in retrieved_docs:
            doc_frequencies[doc_id] = doc_frequencies.get(doc_id, 0) + 1
        total_retrievals += len(retrieved_docs)
    
    if not doc_frequencies:
        return {}
    
    # Compute diversity metrics
    unique_docs_retrieved = len(doc_frequencies)
    avg_frequency = total_retrievals / unique_docs_retrieved if unique_docs_retrieved > 0 else 0
    
    # Coverage ratio
    coverage_ratio = unique_docs_retrieved / corpus_size
    
    # Gini coefficient for diversity
    frequencies = list(doc_frequencies.values())
    frequencies.sort()
    n = len(frequencies)
    if n == 0:
        gini = 0.0
    else:
        cumsum = np.cumsum(frequencies)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    return {
        "unique_docs_retrieved": unique_docs_retrieved,
        "total_retrievals": total_retrievals,
        "avg_doc_frequency": avg_frequency,
        "coverage_ratio": coverage_ratio,
        "gini_diversity": gini,
        "max_doc_frequency": max(frequencies) if frequencies else 0,
        "min_doc_frequency": min(frequencies) if frequencies else 0
    }


def compute_retrieval_effectiveness(retrieved_docs_list: List[List[int]], 
                                  relevant_docs_list: List[Set[int]],
                                  k: int = 5) -> Dict[str, float]:
    """Compute overall retrieval effectiveness metrics."""
    if not retrieved_docs_list:
        return {}
    
    # Compute basic metrics
    basic_metrics = evaluate_retrieval_batch(retrieved_docs_list, relevant_docs_list, [k])
    
    # Compute additional effectiveness metrics
    total_queries = len(retrieved_docs_list)
    successful_queries = 0  # Queries with at least one relevant doc retrieved
    
    for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
        if set(retrieved_docs[:k]) & relevant_docs:
            successful_queries += 1
    
    success_rate = successful_queries / total_queries if total_queries > 0 else 0.0
    
    # Average number of relevant docs retrieved per query
    avg_relevant_retrieved = []
    for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
        relevant_retrieved = len(set(retrieved_docs[:k]) & relevant_docs)
        avg_relevant_retrieved.append(relevant_retrieved)
    
    avg_relevant_per_query = np.mean(avg_relevant_retrieved)
    
    return {
        **basic_metrics,
        "success_rate": success_rate,
        "avg_relevant_per_query": avg_relevant_per_query,
        "total_queries": total_queries,
        "successful_queries": successful_queries
    }
