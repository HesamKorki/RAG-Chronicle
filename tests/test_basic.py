"""Basic tests for RAG Lab framework."""

import pytest
from pathlib import Path

from rag_lab.config import Config
from rag_lab.data import SquadDataset
from rag_lab.retrievers.boolean import BooleanRetriever
from rag_lab.retrievers.tfidf import TfidfRetriever
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.utils.seeds import set_seed
from rag_lab.utils.text import normalize_text, tokenize_text


def test_config_creation():
    """Test configuration creation."""
    config = Config()
    assert config.generator.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.dataset.name == "rajpurkar/squad"
    assert config.seed == 42


def test_text_utils():
    """Test text processing utilities."""
    text = "Hello, World! This is a test."
    normalized = normalize_text(text)
    tokens = tokenize_text(text)
    
    assert "hello" in normalized
    assert "world" in normalized
    assert len(tokens) > 0
    assert "hello" in [t.lower() for t in tokens]


def test_boolean_retriever():
    """Test boolean retriever functionality."""
    retriever = BooleanRetriever()
    
    # Test corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog jumps over the lazy fox.",
        "The lazy fox sleeps in the sun."
    ]
    
    # Build index
    retriever.index(corpus)
    assert retriever.is_indexed()
    assert retriever.get_corpus_size() == 3
    
    # Test retrieval
    results = retriever.retrieve("quick brown", k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_tfidf_retriever():
    """Test TF-IDF retriever functionality."""
    retriever = TfidfRetriever()
    
    # Test corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog jumps over the lazy fox.",
        "The lazy fox sleeps in the sun."
    ]
    
    # Build index
    retriever.index(corpus)
    assert retriever.is_indexed()
    assert retriever.get_corpus_size() == 3
    
    # Test retrieval
    results = retriever.retrieve("quick brown", k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_bm25_retriever():
    """Test BM25 retriever functionality."""
    retriever = BM25Retriever()
    
    # Test corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog jumps over the lazy fox.",
        "The lazy fox sleeps in the sun."
    ]
    
    # Build index
    retriever.index(corpus)
    assert retriever.is_indexed()
    assert retriever.get_corpus_size() == 3
    
    # Test retrieval
    results = retriever.retrieve("quick brown", k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_seed_setting():
    """Test seed setting functionality."""
    set_seed(42)
    # This test mainly ensures the function doesn't raise an exception
    assert True


def test_config_retriever_mapping():
    """Test retriever configuration mapping."""
    config = Config()
    
    # Test getting retriever configs
    boolean_config = config.get_retriever_config("boolean")
    tfidf_config = config.get_retriever_config("tfidf")
    bm25_config = config.get_retriever_config("bm25")
    
    assert boolean_config.k == 5
    assert tfidf_config.k == 5
    assert bm25_config.k == 5


def test_directory_creation():
    """Test directory creation functionality."""
    config = Config()
    config.setup_directories()
    
    # Check that directories exist
    assert config.output.base_dir.exists()
    assert config.output.artifacts_dir.exists()
    assert config.output.indexes_dir.exists()
    assert config.output.logs_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])
