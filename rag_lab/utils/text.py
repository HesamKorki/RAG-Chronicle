"""Text processing utilities for RAG experiments."""

import re
import string
from typing import List, Set


def normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation except for sentence endings
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    
    # Normalize sentence endings
    text = re.sub(r'[\.\!\?]+', '.', text)
    
    return text.strip()


def tokenize_text(text: str, normalize: bool = True) -> List[str]:
    """Tokenize text into words."""
    if normalize:
        text = normalize_text(text)
    
    # Simple word tokenization
    tokens = re.findall(r'\b\w+\b', text)
    
    return tokens


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """Extract n-grams from tokens."""
    if n <= 0 or n > len(tokens):
        return []
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        ngrams.append(ngram)
    
    return ngrams


def extract_ngrams_range(tokens: List[str], min_n: int, max_n: int) -> List[str]:
    """Extract n-grams in a range."""
    all_ngrams = []
    for n in range(min_n, max_n + 1):
        ngrams = extract_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    
    return all_ngrams


def remove_stopwords(tokens: List[str], stopwords: Set[str]) -> List[str]:
    """Remove stopwords from tokens."""
    return [token for token in tokens if token.lower() not in stopwords]


def get_common_stopwords() -> Set[str]:
    """Get a set of common English stopwords."""
    return {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
        'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
        'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
        'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
        'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
        'come', 'made', 'may', 'part'
    }


def clean_text_for_boolean(text: str) -> str:
    """Clean text specifically for boolean retrieval."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_entities(text: str) -> List[str]:
    """Extract potential named entities from text."""
    # Simple entity extraction based on capitalization patterns
    entities = []
    
    # Find capitalized words (potential proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
    entities.extend(capitalized)
    
    # Find multi-word capitalized phrases
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.extend(phrases)
    
    return list(set(entities))


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute simple text similarity using Jaccard similarity."""
    tokens1 = set(tokenize_text(text1))
    tokens2 = set(tokenize_text(text2))
    
    if not tokens1 and not tokens2:
        return 1.0
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0.0


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can break at a reasonable point
        truncated = truncated[:last_space]
    
    return truncated + suffix
