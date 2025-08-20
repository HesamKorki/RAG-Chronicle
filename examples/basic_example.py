#!/usr/bin/env python3
"""
Basic example demonstrating RAG Lab framework usage.

This script shows how to:
1. Load and process SQuAD data
2. Build different types of indexes
3. Run retrieval experiments
4. Evaluate results
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_lab.config import Config
from rag_lab.data import SquadDataset
from rag_lab.retrievers.boolean import BooleanRetriever
from rag_lab.retrievers.tfidf import TfidfRetriever
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.generator.qwen import QwenGenerator
from rag_lab.eval.squad_metrics import evaluate_squad_batch
from rag_lab.utils.seeds import set_seed
from rag_lab.utils.timing import Timer


def main():
    """Run a basic RAG experiment."""
    print("🚀 RAG Lab - Basic Example")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config = Config()
    config.setup_directories()
    
    # Load dataset (using a small subset for demo)
    print("📚 Loading SQuAD dataset...")
    dataset = SquadDataset(config.dataset)
    dataset.load()
    
    # Get a small subset for demonstration
    qa_pairs = dataset.get_qa_pairs()[:10]  # First 10 QA pairs
    corpus = dataset.get_corpus()
    
    print(f"📊 Dataset loaded: {len(qa_pairs)} QA pairs, {len(corpus)} documents")
    
    # Test different retrievers
    retrievers = {
        "boolean": BooleanRetriever(),
        "tfidf": TfidfRetriever(),
        "bm25": BM25Retriever()
    }
    
    results_summary = {}
    
    for retriever_name, retriever in retrievers.items():
        print(f"\n🔍 Testing {retriever_name.upper()} retriever...")
        
        # Build index
        with Timer(f"Building {retriever_name} index"):
            retriever.index(corpus)
        
        # Test retrieval on a few questions
        test_questions = [qa.question for qa in qa_pairs[:3]]
        
        for i, question in enumerate(test_questions):
            print(f"  Question {i+1}: {question[:50]}...")
            
            # Retrieve documents
            retrieved_docs = retriever.retrieve(question, k=3)
            
            # Get passages
            passages = []
            for doc_id, score in retrieved_docs:
                if doc_id < len(corpus):
                    passages.append(corpus[doc_id][:100] + "...")
            
            print(f"    Retrieved {len(passages)} documents")
            for j, passage in enumerate(passages):
                print(f"      Doc {j+1}: {passage}")
        
        # Store results
        results_summary[retriever_name] = {
            "indexed": retriever.is_indexed(),
            "corpus_size": retriever.get_corpus_size(),
            "stats": retriever.get_index_stats()
        }
    
    # Print summary
    print("\n📈 Results Summary")
    print("=" * 50)
    
    for retriever_name, result in results_summary.items():
        print(f"\n{retriever_name.upper()}:")
        print(f"  Indexed: {result['indexed']}")
        print(f"  Corpus size: {result['corpus_size']}")
        if result['stats']:
            print(f"  Stats: {result['stats']}")
    
    print("\n✅ Basic example completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -e .")
    print("2. Build indexes: rag-lab build-index --retriever tfidf")
    print("3. Run experiments: rag-lab run --retriever tfidf --k 5 --max-samples 100")
    print("4. Run sweep: rag-lab sweep --retrievers boolean tfidf bm25 --k 1 3 5")


if __name__ == "__main__":
    main()
