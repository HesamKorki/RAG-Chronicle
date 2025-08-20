"""RAG Lab - A comprehensive RAG experiment framework."""

__version__ = "0.1.0"

from .config import Config
from .data import SquadDataset
from .generator.qwen import QwenGenerator
from .retrievers.base import Retriever

__all__ = ["Config", "SquadDataset", "QwenGenerator", "Retriever"]
