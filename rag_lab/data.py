"""Data loading and processing for RAG experiments."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, load_dataset
from pydantic import BaseModel

from .config import DatasetConfig
from .utils.text import normalize_text, tokenize_text


class QAPair(BaseModel):
    """A question-answer pair with metadata."""
    
    id: str
    question: str
    answer: str
    answers: List[str]
    context: str
    title: str
    is_impossible: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.question = normalize_text(self.question)
        self.answer = normalize_text(self.answer)
        self.answers = [normalize_text(ans) for ans in self.answers]
        self.context = normalize_text(self.context)


class Document(BaseModel):
    """A document with chunks and metadata."""
    
    id: str
    title: str
    text: str
    chunks: List[str]
    chunk_to_doc: Dict[int, int]  # chunk_idx -> doc_idx
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.text = normalize_text(self.text)


class SquadDataset:
    """SQuAD dataset loader and processor."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize the dataset loader."""
        self.config = config
        self.dataset: Optional[Dataset] = None
        self.documents: List[Document] = []
        self.qa_pairs: List[QAPair] = []
        self.doc_to_qa: Dict[int, List[int]] = defaultdict(list)  # doc_idx -> qa_indices
        
    def load(self) -> None:
        """Load the SQuAD dataset."""
        print(f"Loading dataset: {self.config.name}")
        
        # Load from HuggingFace datasets
        self.dataset = load_dataset(
            self.config.name,
            split=self.config.split,
            cache_dir=self.config.cache_dir
        )
        
        # Process into our format
        self._process_dataset()
        
    def _process_dataset(self) -> None:
        """Process the raw dataset into documents and QA pairs."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        # Group by title to create documents
        title_to_paragraphs = defaultdict(list)
        
        for item in self.dataset:
            title = item.get("title", "unknown")
            context = item.get("context", "")
            
            # Create unique paragraph ID
            paragraph_id = f"{title}_{hash(context) % 10000}"
            
            title_to_paragraphs[title].append({
                "id": paragraph_id,
                "context": context,
                "qa_pairs": []
            })
            
            # Add QA pair
            qa_pair = QAPair(
                id=item.get("id", ""),
                question=item.get("question", ""),
                answer=item.get("answers", {}).get("text", [""])[0] if item.get("answers") else "",
                answers=item.get("answers", {}).get("text", [""]) if item.get("answers") else [""],
                context=context,
                title=title,
                is_impossible=item.get("is_impossible", False)
            )
            
            # Find which paragraph this QA pair belongs to
            for para in title_to_paragraphs[title]:
                if para["context"] == context:
                    para["qa_pairs"].append(qa_pair)
                    break
        
        # Create documents
        doc_idx = 0
        for title, paragraphs in title_to_paragraphs.items():
            for para in paragraphs:
                # Create chunks from the paragraph
                chunks = self._create_chunks(para["context"])
                
                document = Document(
                    id=para["id"],
                    title=title,
                    text=para["context"],
                    chunks=chunks,
                    chunk_to_doc={i: doc_idx for i in range(len(chunks))}
                )
                
                self.documents.append(document)
                
                # Add QA pairs and create mapping
                for qa_pair in para["qa_pairs"]:
                    self.qa_pairs.append(qa_pair)
                    self.doc_to_qa[doc_idx].append(len(self.qa_pairs) - 1)
                
                doc_idx += 1
        
        print(f"Processed {len(self.documents)} documents with {len(self.qa_pairs)} QA pairs")
    
    def _create_chunks(self, text: str, chunk_size: int = 512, chunk_stride: int = 256) -> List[str]:
        """Create overlapping chunks from text."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start += chunk_stride
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def get_corpus(self) -> List[str]:
        """Get the corpus as a list of unique document texts."""
        # Use document texts instead of chunks to avoid duplicates
        # Each document represents a unique paragraph from SQuAD
        corpus = []
        seen_texts = set()
        self._corpus_to_doc_mapping = {}  # Maps corpus index to original document index
        self._doc_to_corpus_mapping = {}  # Maps original document index to corpus index
        
        corpus_idx = 0
        for doc_idx, doc in enumerate(self.documents):
            # Use the main document text, not chunks
            if doc.text not in seen_texts:
                corpus.append(doc.text)
                seen_texts.add(doc.text)
                
                # Create bidirectional mapping
                self._corpus_to_doc_mapping[corpus_idx] = doc_idx
                self._doc_to_corpus_mapping[doc_idx] = corpus_idx
                corpus_idx += 1
        
        return corpus
    
    def get_qa_pairs(self) -> List[QAPair]:
        """Get all QA pairs."""
        return self.qa_pairs
    
    def find_ground_truth_doc_id(self, qa_pair: QAPair) -> Optional[int]:
        """Find the document ID that contains the ground truth answer for a QA pair."""
        # Since each document corresponds to a paragraph and each QA pair has the context,
        # we can find the matching document by comparing the context with document text
        for doc_idx, document in enumerate(self.documents):
            if document.text.strip() == qa_pair.context.strip():
                return doc_idx
        return None
    
    def find_ground_truth_corpus_id(self, qa_pair: QAPair) -> Optional[int]:
        """Find the corpus index that contains the ground truth answer for a QA pair."""
        # First find the original document ID
        doc_id = self.find_ground_truth_doc_id(qa_pair)
        if doc_id is None:
            return None
        
        # Convert to corpus index using the mapping
        return self._doc_to_corpus_mapping.get(doc_id, None)
    
    def get_documents(self) -> List[Document]:
        """Get all documents."""
        return self.documents
    
    def get_doc_to_qa_mapping(self) -> Dict[int, List[int]]:
        """Get mapping from document index to QA pair indices."""
        return dict(self.doc_to_qa)
    
    def save_processed(self, output_path: Path) -> None:
        """Save processed dataset to disk."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "chunks": doc.chunks,
                "chunk_to_doc": doc.chunk_to_doc
            })
        
        with open(output_path / "documents.json", "w") as f:
            json.dump(docs_data, f, indent=2)
        
        # Save QA pairs
        qa_data = []
        for qa in self.qa_pairs:
            qa_data.append({
                "id": qa.id,
                "question": qa.question,
                "answer": qa.answer,
                "answers": qa.answers,
                "context": qa.context,
                "title": qa.title,
                "is_impossible": qa.is_impossible
            })
        
        with open(output_path / "qa_pairs.json", "w") as f:
            json.dump(qa_data, f, indent=2)
        
        # Save mapping
        with open(output_path / "doc_to_qa.json", "w") as f:
            json.dump(self.doc_to_qa, f, indent=2)
        
        print(f"Saved processed dataset to {output_path}")
    
    def load_processed(self, input_path: Path) -> None:
        """Load processed dataset from disk."""
        # Load documents
        with open(input_path / "documents.json", "r") as f:
            docs_data = json.load(f)
        
        self.documents = []
        for doc_data in docs_data:
            doc = Document(**doc_data)
            self.documents.append(doc)
        
        # Load QA pairs
        with open(input_path / "qa_pairs.json", "r") as f:
            qa_data = json.load(f)
        
        self.qa_pairs = []
        for qa_data in qa_data:
            qa = QAPair(**qa_data)
            self.qa_pairs.append(qa)
        
        # Load mapping
        with open(input_path / "doc_to_qa.json", "r") as f:
            self.doc_to_qa = defaultdict(list, json.load(f))
        
        print(f"Loaded processed dataset from {input_path}")
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get dataset statistics."""
        total_chunks = sum(len(doc.chunks) for doc in self.documents)
        avg_chunk_length = np.mean([len(chunk) for doc in self.documents for chunk in doc.chunks])
        avg_question_length = np.mean([len(qa.question) for qa in self.qa_pairs])
        avg_answer_length = np.mean([len(qa.answer) for qa in self.qa_pairs])
        
        return {
            "num_documents": len(self.documents),
            "num_qa_pairs": len(self.qa_pairs),
            "total_chunks": total_chunks,
            "avg_chunk_length": avg_chunk_length,
            "avg_question_length": avg_question_length,
            "avg_answer_length": avg_answer_length,
            "impossible_questions": sum(1 for qa in self.qa_pairs if qa.is_impossible)
        }
