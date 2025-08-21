"""Configuration management for RAG experiments."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, Field


def _get_optimal_device() -> str:
    """Get the optimal device based on availability and memory constraints."""
    if torch.backends.mps.is_available():
        # For now, default to CPU for large language models due to MPS memory limitations
        # Users can manually override to "mps" if they have sufficient memory
        print("MPS is available but defaulting to CPU for large model compatibility")
        print("You can override this by setting device='mps' in config if you have sufficient memory")
        return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class DeviceConfig(BaseModel):
    """Device configuration for models."""
    
    device: str = Field(
        default_factory=lambda: _get_optimal_device(),
        description="Device to use for models (mps, cuda, cpu)"
    )
    torch_dtype: str = Field(default="auto", description="Torch dtype for models")


class GeneratorConfig(BaseModel):
    """Configuration for text generation."""
    
    model_name: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens: int = Field(default=32, ge=1, le=512)  # Reduced for better memory efficiency
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = Field(default=42)
    context_token_budget: int = Field(default=1024, ge=512, le=8192)  # Reduced for better memory efficiency
    stop_sequences: List[str] = Field(default_factory=list)
    device: DeviceConfig = Field(default_factory=DeviceConfig)


class RetrieverConfig(BaseModel):
    """Base configuration for retrievers."""
    
    k: int = Field(default=5, ge=1, le=100)
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_stride: int = Field(default=256, ge=0, le=1024)
    max_chars: Optional[int] = Field(default=None, ge=100)


class BooleanRetrieverConfig(RetrieverConfig):
    """Configuration for boolean retrieval."""
    
    normalize_tokens: bool = Field(default=True)
    case_sensitive: bool = Field(default=False)
    min_term_threshold: int = Field(default=3, ge=1, le=10)


class TfidfRetrieverConfig(RetrieverConfig):
    """Configuration for TF-IDF retrieval."""
    
    ngram_range: tuple = Field(default=(1, 2))
    min_df: int = Field(default=2, ge=1)
    max_df: float = Field(default=1.0, ge=0.0, le=1.0)
    norm: str = Field(default="l2", pattern="^(l1|l2)$")


class BM25RetrieverConfig(RetrieverConfig):
    """Configuration for BM25 retrieval."""
    
    k1: float = Field(default=1.5, ge=0.0, le=10.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class DenseRetrieverConfig(RetrieverConfig):
    """Configuration for dense retrieval."""
    
    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    normalize_embeddings: bool = Field(default=True)
    faiss_index_type: str = Field(default="IndexFlatIP")
    batch_size: int = Field(default=32, ge=1, le=256)


class SOTARetrieverConfig(DenseRetrieverConfig):
    """Configuration for SOTA retrieval (dense + reranking)."""
    
    k_rerank: int = Field(default=50, ge=10, le=200)
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_batch_size: int = Field(default=16, ge=1, le=64)


class DatasetConfig(BaseModel):
    """Configuration for datasets."""
    
    name: str = Field(default="rajpurkar/squad")
    split: str = Field(default="validation")
    version: str = Field(default="1.1")
    cache_dir: Optional[str] = Field(default=None)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    
    metrics: List[str] = Field(default=["exact_match", "f1"])
    retrieval_metrics: List[str] = Field(default=["recall@k", "mrr@k", "ndcg@k"])
    k_values: List[int] = Field(default=[1, 3, 5, 10])
    save_predictions: bool = Field(default=True)
    save_plots: bool = Field(default=True)


class OutputConfig(BaseModel):
    """Configuration for output paths and artifacts."""
    
    base_dir: Path = Field(default=Path("runs"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    indexes_dir: Path = Field(default=Path("artifacts/indexes"))
    logs_dir: Path = Field(default=Path("logs"))
    
    def get_run_dir(self, timestamp: str) -> Path:
        """Get directory for a specific run."""
        return self.base_dir / timestamp
    
    def get_index_dir(self, dataset: str, retriever: str) -> Path:
        """Get directory for a specific index."""
        return self.indexes_dir / dataset / retriever


class Config(BaseModel):
    """Main configuration for RAG experiments."""
    
    # Core components
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Retriever-specific configs
    boolean: BooleanRetrieverConfig = Field(default_factory=BooleanRetrieverConfig)
    tfidf: TfidfRetrieverConfig = Field(default_factory=TfidfRetrieverConfig)
    bm25: BM25RetrieverConfig = Field(default_factory=BM25RetrieverConfig)
    dense: DenseRetrieverConfig = Field(default_factory=DenseRetrieverConfig)
    sota: SOTARetrieverConfig = Field(default_factory=SOTARetrieverConfig)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    
    # Reproducibility
    seed: int = Field(default=42)
    
    def get_retriever_config(self, retriever_type: str):
        """Get configuration for a specific retriever type."""
        config_map = {
            "boolean": self.boolean,
            "tfidf": self.tfidf,
            "bm25": self.bm25,
            "dense": self.dense,
            "sota": self.sota,
        }
        if retriever_type not in config_map:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        return config_map[retriever_type]
    
    def setup_directories(self) -> None:
        """Create necessary directories."""
        for path in [self.output.base_dir, self.output.artifacts_dir, 
                    self.output.indexes_dir, self.output.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
