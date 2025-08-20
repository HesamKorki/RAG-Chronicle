"""Dense retriever implementation using sentence transformers and FAISS."""

import numpy as np
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever
from ..utils.io import save_numpy, load_numpy, save_pickle, load_pickle


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers and FAISS."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", 
                 normalize_embeddings: bool = True, faiss_index_type: str = "IndexFlatIP",
                 batch_size: int = 32):
        """Initialize the dense retriever."""
        super().__init__("Dense")
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.faiss_index_type = faiss_index_type
        self.batch_size = batch_size
        
        # Initialize sentence transformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512  # Set reasonable max length
        
        # FAISS index
        self.index = None
        self.embeddings = None
        self.dimension = None
    
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build dense index from corpus."""
        print(f"Building dense index for {len(corpus)} documents...")
        print(f"Using model: {self.model_name}")
        
        self.corpus_size = len(corpus)
        
        # Compute embeddings
        print("Computing embeddings...")
        self.embeddings = self.model.encode(
            corpus,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        self.dimension = self.embeddings.shape[1]
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        # Create FAISS index
        print(f"Creating FAISS index: {self.faiss_index_type}")
        if self.faiss_index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.faiss_index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.faiss_index_type}")
        
        # Add vectors to index
        self.index.add(self.embeddings.astype('float32'))
        
        self.indexed = True
        print(f"Built dense index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using dense similarity."""
        if not self.indexed:
            raise ValueError("Index not built. Call index() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return (doc_id, score) pairs
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for invalid indices
                results.append((int(idx), float(score)))
        
        return results
    
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss_index.bin"))
        
        # Save embeddings
        save_numpy(self.embeddings, path / "embeddings.npy")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings,
            'faiss_index_type': self.faiss_index_type,
            'batch_size': self.batch_size,
            'corpus_size': self.corpus_size,
            'dimension': self.dimension
        }
        save_pickle(metadata, path / "dense_metadata.pkl")
        
        print(f"Saved dense index to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        # Load FAISS index
        index_file = path / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        # Load embeddings
        embeddings_file = path / "embeddings.npy"
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        self.embeddings = load_numpy(embeddings_file)
        
        # Load metadata
        metadata_file = path / "dense_metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        metadata = load_pickle(metadata_file)
        self.model_name = metadata['model_name']
        self.normalize_embeddings = metadata['normalize_embeddings']
        self.faiss_index_type = metadata['faiss_index_type']
        self.batch_size = metadata['batch_size']
        self.corpus_size = metadata['corpus_size']
        self.dimension = metadata['dimension']
        
        # Reinitialize model if needed
        if not hasattr(self, 'model') or self.model is None:
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = 512
        
        self.indexed = True
        print(f"Loaded dense index from {path}")
    
    def get_document_embedding(self, doc_id: int) -> np.ndarray:
        """Get embedding for a specific document."""
        if not self.indexed or doc_id >= self.corpus_size:
            return np.array([])
        
        return self.embeddings[doc_id]
    
    def get_similar_documents(self, doc_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Find documents similar to a given document."""
        if not self.indexed or doc_id >= self.corpus_size:
            return []
        
        # Get document embedding
        doc_embedding = self.embeddings[doc_id:doc_id+1].astype('float32')
        
        # Search in FAISS index
        scores, indices = self.index.search(doc_embedding, k + 1)  # +1 to exclude self
        
        # Return results (excluding self)
        results = []
        for score, idx in zip(scores[0][1:], indices[0][1:]):  # Skip first result (self)
            if idx != -1:
                results.append((int(idx), float(score)))
        
        return results
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        if not self.indexed:
            return {}
        
        return {
            'num_documents': self.corpus_size,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'faiss_index_type': self.faiss_index_type,
            'normalize_embeddings': self.normalize_embeddings,
            'index_size': self.index.ntotal,
            'embeddings_shape': self.embeddings.shape
        }
    
    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for a query."""
        return self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings
        )[0]
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings
        )
