#!/usr/bin/env python3
"""
Simple script to test Qwen generation independently.
Ask questions directly to the model without any retrieval.
"""

import sys
from rag_lab.config import Config
from rag_lab.generator import QwenGenerator


def test_generation():
    """Interactive test of the Qwen generator."""
    print("🤖 RAG Lab - Qwen Generator Test")
    print("=" * 50)
    
    # Load default config
    config = Config()
    print(f"Device: {config.generator.device.device}")
    print(f"Model: {config.generator.model_name}")
    print(f"Max tokens: {config.generator.max_new_tokens}")
    print("-" * 50)
    
    # Initialize generator
    print("Loading model...")
    generator = QwenGenerator(config.generator)
    print("Model loaded successfully!")
    print("-" * 50)
    
    print("You can now ask questions directly to the model.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user question
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.\n")
                continue
            
            print(f"\n🤔 Thinking...")
            
            # Generate answer without any passages (direct question answering)
            result = generator.generate(
                question=question,
                passages=[],  # No retrieved passages
                doc_ids=[],
                k=0
            )
            
            print(f"🤖 Answer: {result['answer']}")
            print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
            print(f"📊 Tokens generated: {result['tokens_generated']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Try asking another question.\n")


def test_with_context():
    """Test generation with some example context."""
    print("🤖 RAG Lab - Qwen Generator Test (With Context)")
    print("=" * 60)
    
    # Load default config
    config = Config()
    generator = QwenGenerator(config.generator)
    
    # Example context
    example_passages = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
        "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. It was the world's tallest man-made structure from 1889 to 1930.",
        "The Eiffel Tower was built between 1887 and 1889 as the entrance to the 1889 World's Fair."
    ]
    
    questions = [
        "How tall is the Eiffel Tower?",
        "Who designed the Eiffel Tower?",
        "When was the Eiffel Tower built?",
        "Where is the Eiffel Tower located?"
    ]
    
    print("Testing with example context about the Eiffel Tower...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        
        result = generator.generate(
            question=question,
            passages=example_passages,
            doc_ids=[1, 2, 3],
            k=3
        )
        
        print(f"Answer: {result['answer']}")
        print(f"Generation time: {result['generation_time']:.2f}s")
        print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--context":
        test_with_context()
    else:
        test_generation()
