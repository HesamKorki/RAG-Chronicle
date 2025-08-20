"""TF-IDF retriever implementation."""

import numpy as np
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever
from ..utils.io import save_pickle, load_pickle


class TfidfRetriever(BaseRetriever):
    """TF-IDF retriever using scikit-learn."""
    
    def __init__(self, ngram_range: tuple = (1, 2), min_df: int = 2, 
                 max_df: float = 1.0, norm: str = 'l2'):
        """Initialize the TF-IDF retriever."""
        super().__init__("TF-IDF")
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            norm=norm,
            lowercase=True,
            stop_words='english'
        )
        
        self.tfidf_matrix = None
        self.feature_names = None
    
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build TF-IDF index from corpus."""
        print(f"Building TF-IDF index for {len(corpus)} documents...")
        
        # Fit and transform the corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.corpus_size = len(corpus)
        
        self.indexed = True
        print(f"Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")
    
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using TF-IDF similarity."""
        if not self.indexed:
            raise ValueError("Index not built. Call index() first.")
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return (doc_id, score) pairs
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        save_pickle(self.vectorizer, path / "tfidf_vectorizer.pkl")
        
        # Save TF-IDF matrix
        save_pickle(self.tfidf_matrix, path / "tfidf_matrix.pkl")
        
        # Save metadata
        metadata = {
            'corpus_size': self.corpus_size,
            'feature_names': self.feature_names,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'norm': self.norm
        }
        save_pickle(metadata, path / "tfidf_metadata.pkl")
        
        print(f"Saved TF-IDF index to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        # Load vectorizer
        vectorizer_file = path / "tfidf_vectorizer.pkl"
        if not vectorizer_file.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")
        
        self.vectorizer = load_pickle(vectorizer_file)
        
        # Load TF-IDF matrix
        matrix_file = path / "tfidf_matrix.pkl"
        if not matrix_file.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_file}")
        
        self.tfidf_matrix = load_pickle(matrix_file)
        
        # Load metadata
        metadata_file = path / "tfidf_metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        metadata = load_pickle(metadata_file)
        self.corpus_size = metadata['corpus_size']
        self.feature_names = metadata['feature_names']
        self.ngram_range = metadata['ngram_range']
        self.min_df = metadata['min_df']
        self.max_df = metadata['max_df']
        self.norm = metadata['norm']
        
        self.indexed = True
        print(f"Loaded TF-IDF index from {path}")
    
    def get_feature_importance(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most important features for a query."""
        if not self.indexed:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Get feature names and their importance scores
        feature_scores = list(zip(self.feature_names, query_vector.toarray()[0]))
        
        # Sort by importance and return top-n
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(feature, score) for feature, score in feature_scores[:top_n] if score > 0]
    
    def get_document_vector(self, doc_id: int) -> np.ndarray:
        """Get TF-IDF vector for a specific document."""
        if not self.indexed or doc_id >= self.corpus_size:
            return np.array([])
        
        return self.tfidf_matrix[doc_id].toarray().flatten()
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        if not self.indexed:
            return {}
        
        # Calculate sparsity
        total_elements = self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]
        non_zero_elements = self.tfidf_matrix.nnz
        sparsity = 1 - (non_zero_elements / total_elements)
        
        # Calculate average document length
        doc_lengths = np.sum(self.tfidf_matrix > 0, axis=1).A1
        avg_doc_length = np.mean(doc_lengths)
        
        return {
            'num_documents': self.tfidf_matrix.shape[0],
            'num_features': self.tfidf_matrix.shape[1],
            'sparsity': sparsity,
            'avg_doc_length': avg_doc_length,
            'total_elements': total_elements,
            'non_zero_elements': non_zero_elements
        }
    
    def get_similar_documents(self, doc_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Find documents similar to a given document."""
        if not self.indexed or doc_id >= self.corpus_size:
            return []
        
        # Get document vector
        doc_vector = self.tfidf_matrix[doc_id:doc_id+1]
        
        # Compute similarities with all documents
        similarities = cosine_similarity(doc_vector, self.tfidf_matrix).flatten()
        
        # Get top-k similar documents (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:k+1]  # Skip first (self)
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
