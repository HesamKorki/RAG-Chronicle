"""Boolean retriever implementation."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .base import BaseRetriever
from ..utils.io import save_pickle, load_pickle
from ..utils.text import clean_text_for_boolean, tokenize_text


class BooleanRetriever(BaseRetriever):
    """Boolean retriever with AND/OR/NOT operations."""
    
    def __init__(self, normalize_tokens: bool = True, case_sensitive: bool = False):
        """Initialize the boolean retriever."""
        super().__init__("Boolean")
        self.normalize_tokens = normalize_tokens
        self.case_sensitive = case_sensitive
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        self.corpus: List[str] = []
    
    def index(self, corpus: List[str], **kwargs) -> None:
        """Build inverted index from corpus."""
        print(f"Building boolean index for {len(corpus)} documents...")
        
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.inverted_index.clear()
        
        for doc_idx, document in enumerate(corpus):
            # Clean and tokenize document
            if self.normalize_tokens:
                doc_text = clean_text_for_boolean(document)
            else:
                doc_text = document
            
            tokens = tokenize_text(doc_text, normalize=False)
            
            # Add to inverted index
            for token in tokens:
                if not self.case_sensitive:
                    token = token.lower()
                self.inverted_index[token].add(doc_idx)
        
        self.indexed = True
        print(f"Built index with {len(self.inverted_index)} unique terms")
    
    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using boolean operations."""
        if not self.indexed:
            raise ValueError("Index not built. Call index() first.")
        
        # Parse boolean query
        doc_ids = self._evaluate_boolean_query(query)
        
        # Convert to list of tuples with dummy scores
        results = [(doc_id, 1.0) for doc_id in doc_ids]
        
        # Limit to k results
        return results[:k]
    
    def _evaluate_boolean_query(self, query: str) -> Set[int]:
        """Evaluate a boolean query with AND/OR/NOT operations."""
        # Clean query
        query = query.strip()
        
        # Handle simple queries (no operators)
        if not any(op in query.upper() for op in ['AND', 'OR', 'NOT']):
            return self._get_docs_for_term(query)
        
        # Parse complex queries
        return self._parse_boolean_expression(query)
    
    def _parse_boolean_expression(self, expression: str) -> Set[int]:
        """Parse and evaluate boolean expression."""
        # Simple implementation - can be extended for more complex parsing
        expression = expression.upper()
        
        # Handle NOT
        if expression.startswith('NOT '):
            term = expression[4:].strip()
            all_docs = set(range(self.corpus_size))
            term_docs = self._get_docs_for_term(term)
            return all_docs - term_docs
        
        # Handle AND
        if ' AND ' in expression:
            terms = [term.strip() for term in expression.split(' AND ')]
            result = self._get_docs_for_term(terms[0])
            for term in terms[1:]:
                result = result.intersection(self._get_docs_for_term(term))
            return result
        
        # Handle OR
        if ' OR ' in expression:
            terms = [term.strip() for term in expression.split(' OR ')]
            result = set()
            for term in terms:
                result = result.union(self._get_docs_for_term(term))
            return result
        
        # Fallback to simple term lookup
        return self._get_docs_for_term(expression)
    
    def _get_docs_for_term(self, term: str) -> Set[int]:
        """Get document IDs for a single term."""
        if not self.case_sensitive:
            term = term.lower()
        
        # Clean term if needed
        if self.normalize_tokens:
            term = clean_text_for_boolean(term)
            # Extract tokens and get intersection
            tokens = tokenize_text(term, normalize=False)
            if not tokens:
                return set()
            
            result = self.inverted_index.get(tokens[0], set())
            for token in tokens[1:]:
                result = result.intersection(self.inverted_index.get(token, set()))
            return result
        else:
            return self.inverted_index.get(term, set())
    
    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'inverted_index': dict(self.inverted_index),
            'corpus_size': self.corpus_size,
            'normalize_tokens': self.normalize_tokens,
            'case_sensitive': self.case_sensitive
        }
        
        save_pickle(index_data, path / "boolean_index.pkl")
        print(f"Saved boolean index to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        index_file = path / "boolean_index.pkl"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        index_data = load_pickle(index_file)
        
        self.inverted_index = defaultdict(set, index_data['inverted_index'])
        self.corpus_size = index_data['corpus_size']
        self.normalize_tokens = index_data['normalize_tokens']
        self.case_sensitive = index_data['case_sensitive']
        self.indexed = True
        
        print(f"Loaded boolean index from {path}")
    
    def get_index_stats(self) -> Dict[str, int]:
        """Get statistics about the index."""
        if not self.indexed:
            return {}
        
        term_frequencies = [len(docs) for docs in self.inverted_index.values()]
        
        return {
            'num_terms': len(self.inverted_index),
            'num_documents': self.corpus_size,
            'avg_term_frequency': sum(term_frequencies) / len(term_frequencies) if term_frequencies else 0,
            'max_term_frequency': max(term_frequencies) if term_frequencies else 0,
            'min_term_frequency': min(term_frequencies) if term_frequencies else 0
        }
    
    def search_terms(self, term: str) -> List[str]:
        """Search for terms in the index (useful for autocomplete)."""
        if not self.indexed:
            return []
        
        if not self.case_sensitive:
            term = term.lower()
        
        matching_terms = []
        for index_term in self.inverted_index.keys():
            if term in index_term:
                matching_terms.append(index_term)
        
        return sorted(matching_terms)
