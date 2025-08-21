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
    
    def __init__(self, normalize_tokens: bool = True, case_sensitive: bool = False, min_term_threshold: int = 3):
        """Initialize the boolean retriever."""
        super().__init__("Boolean")
        self.normalize_tokens = normalize_tokens
        self.case_sensitive = case_sensitive
        self.min_term_threshold = min_term_threshold  # Minimum non-stopword terms required
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
        
        # TRUE Boolean retrieval: Binary relevance only (1.0 or 0.0)
        # All matching documents get equal score - no ranking by relevance
        results = [(doc_id, 1.0) for doc_id in doc_ids]
        
        # Return first k results (no meaningful ranking in pure Boolean)
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
        """Get document IDs for a single term or natural language query."""
        if not self.case_sensitive:
            term = term.lower()
        
        # Clean term if needed
        if self.normalize_tokens:
            term = clean_text_for_boolean(term)
            # Extract tokens
            tokens = tokenize_text(term, normalize=False)
            if not tokens:
                return set()
            
            # Filter out common stopwords for better natural language handling
            stopwords = {'what', 'who', 'where', 'when', 'why', 'how', 'the', 'a', 'an', 
                        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'}
            
            meaningful_tokens = [token for token in tokens if token.lower() not in stopwords]
            
            # If no meaningful tokens, fall back to all tokens
            if not meaningful_tokens:
                meaningful_tokens = tokens
            
            # Apply threshold: only return documents that contain at least min_term_threshold meaningful terms
            if len(meaningful_tokens) >= self.min_term_threshold:
                candidate_docs = set()
                for token in meaningful_tokens:
                    candidate_docs = candidate_docs.union(self.inverted_index.get(token, set()))
                
                # Filter documents that meet the threshold
                result = set()
                for doc_id in candidate_docs:
                    doc_text = self.corpus[doc_id].lower() if not self.case_sensitive else self.corpus[doc_id]
                    doc_tokens = set(tokenize_text(doc_text, normalize=False))
                    doc_tokens = {token.lower() for token in doc_tokens}
                    
                    # Count how many meaningful query terms appear in this document
                    matching_terms = sum(1 for token in meaningful_tokens if token.lower() in doc_tokens)
                    
                    # Only include if it meets the threshold
                    if matching_terms >= self.min_term_threshold:
                        result.add(doc_id)
                
                return result
            else:
                # If query has fewer than threshold terms, use OR logic without threshold
                result = set()
                for token in meaningful_tokens:
                    result = result.union(self.inverted_index.get(token, set()))
                return result
        else:
            return self.inverted_index.get(term, set())
    

    def save_index(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'inverted_index': dict(self.inverted_index),
            'corpus': self.corpus,
            'corpus_size': self.corpus_size,
            'normalize_tokens': self.normalize_tokens,
            'case_sensitive': self.case_sensitive,
            'min_term_threshold': self.min_term_threshold
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
        self.corpus = index_data['corpus']
        self.corpus_size = index_data['corpus_size']
        self.normalize_tokens = index_data['normalize_tokens']
        self.case_sensitive = index_data['case_sensitive']
        self.min_term_threshold = index_data.get('min_term_threshold', 3)  # Default to 3 for backward compatibility
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
