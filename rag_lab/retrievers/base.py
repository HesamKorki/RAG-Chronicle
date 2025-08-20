"""Base retriever interface and abstract base class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Protocol, Tuple


class Retriever(Protocol):
    """Protocol for retriever implementations."""
    
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build index from corpus."""
        ...
    
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for query."""
        ...
    
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        ...
    
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        ...


class BaseRetriever(ABC):
    """Abstract base class for retriever implementations."""
    
    def __init__(self, name: str):
        """Initialize the retriever."""
        self.name = name
        self.indexed = False
        self.corpus_size = 0
    
    @abstractmethod
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build index from corpus."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for query."""
        pass
    
    @abstractmethod
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        pass
    
    def is_indexed(self) -> bool:
        """Check if the retriever has been indexed."""
        return self.indexed
    
    def get_corpus_size(self) -> int:
        """Get the size of the indexed corpus."""
        return self.corpus_size
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}Retriever(corpus_size={self.corpus_size}, indexed={self.indexed})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
