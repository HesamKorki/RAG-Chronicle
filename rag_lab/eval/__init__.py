"""Evaluation metrics for RAG experiments."""

from .squad_metrics import (
    normalize_answer,
    f1_score,
    exact_match_score,
    evaluate_squad_answers,
    evaluate_squad_batch,
    compute_confidence_metrics,
    analyze_error_types
)

from .retrieval_metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_hit_at_k,
    evaluate_retrieval_batch,
    compute_retrieval_latency,
    compute_context_usage_stats,
    analyze_retrieval_diversity,
    compute_retrieval_effectiveness
)

__all__ = [
    # SQuAD metrics
    "normalize_answer",
    "f1_score", 
    "exact_match_score",
    "evaluate_squad_answers",
    "evaluate_squad_batch",
    "compute_confidence_metrics",
    "analyze_error_types",
    
    # Retrieval metrics
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr_at_k",
    "compute_ndcg_at_k",
    "compute_hit_at_k",
    "evaluate_retrieval_batch",
    "compute_retrieval_latency",
    "compute_context_usage_stats",
    "analyze_retrieval_diversity",
    "compute_retrieval_effectiveness"
]
