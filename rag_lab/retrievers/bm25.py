"""BM25 retriever implementation."""

import numpy as np
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .base import BaseRetriever
from ..utils.io import save_pickle, load_pickle
from ..utils.text import tokenize_text


class BM25Retriever(BaseRetriever):
    """BM25 retriever using rank-bm25."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize the BM25 retriever."""
        super().__init__("BM25")
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus_tokens = []
    
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build BM25 index from corpus."""
        print(f"Building BM25 index for {len(corpus)} documents...")
        
        # Tokenize corpus
        self.corpus_tokens = [tokenize_text(doc) for doc in corpus]
        self.corpus_size = len(corpus)
        
        # Create BM25 model
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        
        self.indexed = True
        print(f"Built BM25 index with k1={self.k1}, b={self.b}")
    
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using BM25 scoring."""
        if not self.indexed:
            raise ValueError("Index not built. Call index() first.")
        
        # Tokenize query
        query_tokens = tokenize_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return (doc_id, score) pairs
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 model
        save_pickle(self.bm25, path / "bm25_model.pkl")
        
        # Save metadata
        metadata = {
            'corpus_size': self.corpus_size,
            'corpus_tokens': self.corpus_tokens,
            'k1': self.k1,
            'b': self.b
        }
        save_pickle(metadata, path / "bm25_metadata.pkl")
        
        print(f"Saved BM25 index to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        # Load BM25 model
        model_file = path / "bm25_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"BM25 model file not found: {model_file}")
        
        self.bm25 = load_pickle(model_file)
        
        # Load metadata
        metadata_file = path / "bm25_metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        metadata = load_pickle(metadata_file)
        self.corpus_size = metadata['corpus_size']
        self.corpus_tokens = metadata['corpus_tokens']
        self.k1 = metadata['k1']
        self.b = metadata['b']
        
        self.indexed = True
        print(f"Loaded BM25 index from {path}")
    
    def get_document_score(self, query: str, doc_id: int) -> float:
        """Get BM25 score for a specific document."""
        if not self.indexed or doc_id >= self.corpus_size:
            return 0.0
        
        query_tokens = tokenize_text(query)
        scores = self.bm25.get_scores(query_tokens)
        
        return float(scores[doc_id])
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        if not self.indexed:
            return {}
        
        # Calculate average document length
        doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        avg_doc_length = np.mean(doc_lengths)
        
        # Calculate vocabulary size
        all_tokens = set()
        for tokens in self.corpus_tokens:
            all_tokens.update(tokens)
        vocab_size = len(all_tokens)
        
        return {
            'num_documents': self.corpus_size,
            'vocabulary_size': vocab_size,
            'avg_doc_length': avg_doc_length,
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths),
            'k1': self.k1,
            'b': self.b
        }
    
    def get_term_frequency(self, term: str) -> int:
        """Get document frequency of a term."""
        if not self.indexed:
            return 0
        
        count = 0
        for tokens in self.corpus_tokens:
            if term in tokens:
                count += 1
        
        return count
    
    def get_top_terms(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """Get most frequent terms in the corpus."""
        if not self.indexed:
            return []
        
        term_freq = {}
        for tokens in self.corpus_tokens:
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_terms[:top_n]
    
    def get_document_tokens(self, doc_id: int) -> List[str]:
        """Get tokens for a specific document."""
        if not self.indexed or doc_id >= self.corpus_size:
            return []
        
        return self.corpus_tokens[doc_id]
