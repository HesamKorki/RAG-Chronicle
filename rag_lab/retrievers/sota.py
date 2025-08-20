"""SOTA retriever implementation with dense retrieval + cross-encoder reranking."""

import numpy as np
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from .dense import DenseRetriever


class SOTARetriever(DenseRetriever):
    """SOTA retriever combining dense retrieval with cross-encoder reranking."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", 
                 normalize_embeddings: bool = True, faiss_index_type: str = "IndexFlatIP",
                 batch_size: int = 32, k_rerank: int = 50,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v-2",
                 rerank_batch_size: int = 16):
        """Initialize the SOTA retriever."""
        super().__init__(model_name, normalize_embeddings, faiss_index_type, batch_size)
        self.name = "SOTA"
        
        self.k_rerank = k_rerank
        self.cross_encoder_model = cross_encoder_model
        self.rerank_batch_size = rerank_batch_size
        
        # Initialize cross-encoder for reranking
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Store corpus for reranking
        self.corpus = []
    
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build index and store corpus for reranking."""
        # Store corpus for reranking
        self.corpus = corpus
        # Call parent index method
        super().index(corpus, **kwargs)
    
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using dense retrieval + cross-encoder reranking."""
        if not self.indexed:
            raise ValueError("Index not built. Call index() first.")
        
        # First stage: dense retrieval
        initial_results = super().retrieve(query, self.k_rerank)
        
        if not initial_results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = []
        doc_ids = []
        for doc_id, _ in initial_results:
            pairs.append([query, self.corpus[doc_id]])
            doc_ids.append(doc_id)
        
        # Second stage: cross-encoder reranking
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False
        )
        
        # Combine doc_ids with reranked scores
        reranked_results = list(zip(doc_ids, scores))
        
        # Sort by cross-encoder scores
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return reranked_results[:k]
    
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        # Save dense index components
        super().save_index(path)
        
        # Save SOTA-specific metadata
        sota_metadata = {
            'k_rerank': self.k_rerank,
            'cross_encoder_model': self.cross_encoder_model,
            'rerank_batch_size': self.rerank_batch_size,
            'corpus': self.corpus
        }
        
        from ..utils.io import save_pickle
        save_pickle(sota_metadata, path / "sota_metadata.pkl")
        
        print(f"Saved SOTA index to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        # Load dense index components
        super().load_index(path)
        
        # Load SOTA-specific metadata
        sota_metadata_file = path / "sota_metadata.pkl"
        if sota_metadata_file.exists():
            from ..utils.io import load_pickle
            sota_metadata = load_pickle(sota_metadata_file)
            
            self.k_rerank = sota_metadata['k_rerank']
            self.cross_encoder_model = sota_metadata['cross_encoder_model']
            self.rerank_batch_size = sota_metadata['rerank_batch_size']
            self.corpus = sota_metadata.get('corpus', [])
            
            # Reinitialize cross-encoder
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
        
        print(f"Loaded SOTA index from {path}")
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        stats = super().get_index_stats()
        stats.update({
            'k_rerank': self.k_rerank,
            'cross_encoder_model': self.cross_encoder_model,
            'rerank_batch_size': self.rerank_batch_size
        })
        return stats
    
    def retrieve_with_scores(self, query: str, k: int) -> Tuple[List[Tuple[int, float]], List[float]]:
        """Retrieve documents and return both dense and reranked scores."""
        if not self.indexed:
            raise ValueError("Index not built. Call index() first.")
        
        # First stage: dense retrieval
        initial_results = super().retrieve(query, self.k_rerank)
        
        if not initial_results:
            return [], []
        
        # Prepare pairs for cross-encoder
        pairs = []
        doc_ids = []
        dense_scores = []
        for doc_id, dense_score in initial_results:
            pairs.append([query, self.corpus[doc_id]])
            doc_ids.append(doc_id)
            dense_scores.append(dense_score)
        
        # Second stage: cross-encoder reranking
        rerank_scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False
        )
        
        # Combine all information
        combined_results = list(zip(doc_ids, dense_scores, rerank_scores))
        
        # Sort by cross-encoder scores
        combined_results.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k results
        final_results = [(doc_id, rerank_score) for doc_id, _, rerank_score in combined_results[:k]]
        final_dense_scores = [dense_score for _, dense_score, _ in combined_results[:k]]
        
        return final_results, final_dense_scores
