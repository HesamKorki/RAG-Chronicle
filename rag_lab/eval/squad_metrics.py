"""SQuAD evaluation metrics for RAG experiments."""

import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    import string
    
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score between prediction and ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate_squad_answers(predictions: List[str], ground_truths: List[List[str]], 
                         is_impossible: Optional[List[bool]] = None) -> Dict[str, float]:
    """Evaluate SQuAD-style answers."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions must match number of ground truth sets")
    
    exact_matches = []
    f1_scores = []
    
    for i, (prediction, ground_truth_list) in enumerate(zip(predictions, ground_truths)):
        # Handle impossible questions
        if is_impossible and is_impossible[i]:
            # For impossible questions, check if prediction indicates "don't know"
            pred_lower = prediction.lower().strip()
            if any(phrase in pred_lower for phrase in ["i don't know", "don't know", "cannot answer", "no answer"]):
                exact_matches.append(1.0)
                f1_scores.append(1.0)
            else:
                exact_matches.append(0.0)
                f1_scores.append(0.0)
            continue
        
        # For answerable questions, compute scores against all ground truth answers
        best_f1 = 0.0
        best_exact_match = 0.0
        
        for ground_truth in ground_truth_list:
            f1 = f1_score(prediction, ground_truth)
            exact_match = exact_match_score(prediction, ground_truth)
            
            best_f1 = max(best_f1, f1)
            best_exact_match = max(best_exact_match, exact_match)
        
        f1_scores.append(best_f1)
        exact_matches.append(best_exact_match)
    
    return {
        "exact_match": np.mean(exact_matches),
        "f1": np.mean(f1_scores),
        "exact_match_std": np.std(exact_matches),
        "f1_std": np.std(f1_scores)
    }


def evaluate_squad_batch(results: List[Dict[str, Any]], qa_pairs: List[Any]) -> Dict[str, float]:
    """Evaluate a batch of SQuAD results."""
    predictions = []
    ground_truths = []
    is_impossible = []
    
    for result, qa_pair in zip(results, qa_pairs):
        predictions.append(result["answer"])
        ground_truths.append(qa_pair.answers)
        is_impossible.append(qa_pair.is_impossible)
    
    return evaluate_squad_answers(predictions, ground_truths, is_impossible)


def compute_confidence_metrics(predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, float]:
    """Compute confidence-related metrics."""
    # Simple confidence based on answer length and presence of uncertainty words
    uncertainty_words = ["maybe", "perhaps", "possibly", "i think", "i believe", "might", "could", "don't know"]
    
    confidence_scores = []
    for prediction in predictions:
        pred_lower = prediction.lower()
        
        # Check for uncertainty words
        has_uncertainty = any(word in pred_lower for word in uncertainty_words)
        
        # Simple confidence based on length and uncertainty
        if has_uncertainty:
            confidence = 0.3
        elif len(prediction.strip()) < 5:
            confidence = 0.5
        else:
            confidence = 0.8
        
        confidence_scores.append(confidence)
    
    return {
        "avg_confidence": np.mean(confidence_scores),
        "confidence_std": np.std(confidence_scores)
    }


def analyze_error_types(predictions: List[str], ground_truths: List[List[str]], 
                       is_impossible: Optional[List[bool]] = None) -> Dict[str, Any]:
    """Analyze types of errors made by the model."""
    error_analysis = {
        "no_answer_given": 0,
        "wrong_answer": 0,
        "partial_answer": 0,
        "hallucination": 0,
        "impossible_handled_correctly": 0,
        "impossible_handled_incorrectly": 0
    }
    
    for i, (prediction, ground_truth_list) in enumerate(zip(predictions, ground_truths)):
        pred_lower = prediction.lower().strip()
        
        # Handle impossible questions
        if is_impossible and is_impossible[i]:
            if any(phrase in pred_lower for phrase in ["i don't know", "don't know", "cannot answer"]):
                error_analysis["impossible_handled_correctly"] += 1
            else:
                error_analysis["impossible_handled_incorrectly"] += 1
            continue
        
        # Check for no answer
        if not prediction.strip() or pred_lower in ["", "i don't know", "don't know"]:
            error_analysis["no_answer_given"] += 1
            continue
        
        # Check for exact match
        exact_match = any(exact_match_score(prediction, gt) for gt in ground_truth_list)
        if exact_match:
            continue
        
        # Check for partial match (F1 > 0.5)
        best_f1 = max(f1_score(prediction, gt) for gt in ground_truth_list)
        if best_f1 > 0.5:
            error_analysis["partial_answer"] += 1
        else:
            # Check if it's a hallucination (contains words not in ground truth)
            gt_words = set()
            for gt in ground_truth_list:
                gt_words.update(normalize_answer(gt).split())
            
            pred_words = set(normalize_answer(prediction).split())
            overlap = len(pred_words & gt_words)
            
            if overlap == 0:
                error_analysis["hallucination"] += 1
            else:
                error_analysis["wrong_answer"] += 1
    
    # Convert to percentages
    total = len(predictions)
    for key in error_analysis:
        error_analysis[key] = error_analysis[key] / total * 100
    
    return error_analysis
