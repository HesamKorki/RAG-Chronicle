#!/bin/bash

# RAG Lab Installation Script
# This script helps install the RAG Lab framework and its dependencies using uv

set -e  # Exit on any error

echo "🚀 RAG Lab - Installation Script (using uv)"
echo "============================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv is available"

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv --python 3.10

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install the package in development mode
echo "📦 Installing RAG Lab in development mode..."
uv pip install -e .

# Install additional dependencies for development
echo "📦 Installing development dependencies..."
uv pip install pytest pytest-cov black isort flake8 mypy jupyter ipykernel

# Check for FAISS installation
echo "🔍 Checking FAISS installation..."
if python -c "import faiss" 2>/dev/null; then
    echo "✅ FAISS is already installed"
else
    echo "📦 Installing FAISS..."
    uv pip install faiss-cpu
fi

# Check for PyTorch and MPS support
echo "🔍 Checking PyTorch and MPS support..."
if python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch is installed"
    
    # Check MPS availability
    if python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        echo "✅ MPS (Apple Silicon) support is available"
    else
        echo "⚠️  MPS not available (this is normal on non-Apple Silicon Macs)"
    fi
else
    echo "❌ PyTorch not found. Installing..."
    uv pip install torch
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p runs artifacts/indexes logs notebooks tests

# Run basic tests
echo "🧪 Running basic tests..."
if python -m pytest tests/test_basic.py -v; then
    echo "✅ Basic tests passed"
else
    echo "⚠️  Some tests failed, but installation may still work"
fi

# Create a simple test script
echo "📝 Creating test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify RAG Lab installation."""

try:
    from rag_lab.config import Config
    from rag_lab.data import SquadDataset
    from rag_lab.retrievers.boolean import BooleanRetriever
    from rag_lab.retrievers.tfidf import TfidfRetriever
    from rag_lab.retrievers.bm25 import BM25Retriever
    from rag_lab.utils.seeds import set_seed
    from rag_lab.utils.text import normalize_text
    
    print("✅ All imports successful!")
    
    # Test basic functionality
    config = Config()
    print(f"✅ Config loaded: {config.generator.model_name}")
    
    set_seed(42)
    print("✅ Seed setting works")
    
    text = "Hello, World!"
    normalized = normalize_text(text)
    print(f"✅ Text normalization works: '{text}' -> '{normalized}'")
    
    # Test retriever creation
    boolean_retriever = BooleanRetriever()
    tfidf_retriever = TfidfRetriever()
    bm25_retriever = BM25Retriever()
    print("✅ All retrievers created successfully")
    
    print("\n🎉 Installation test completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment: source .venv/bin/activate")
    print("2. Run: python examples/basic_example.py")
    print("3. Build index: rag-lab build-index --retriever tfidf")
    print("4. Run experiment: rag-lab run --retriever tfidf --k 5 --max-samples 100")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF

# Make test script executable
chmod +x test_installation.py

# Run installation test
echo "🧪 Running installation test..."
if python test_installation.py; then
    echo "✅ Installation test passed"
else
    echo "❌ Installation test failed"
    exit 1
fi

# Create activation script
echo "📝 Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Script to activate the RAG Lab virtual environment

if [ -f ".venv/bin/activate" ]; then
    echo "🔧 Activating RAG Lab virtual environment..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated!"
    echo "📚 You can now run:"
    echo "   - python examples/basic_example.py"
    echo "   - rag-lab build-index --retriever tfidf"
    echo "   - rag-lab run --retriever tfidf --k 5 --max-samples 100"
else
    echo "❌ Virtual environment not found. Run install.sh first."
    exit 1
fi
EOF

chmod +x activate_env.sh

echo ""
echo "🎉 RAG Lab installation completed successfully!"
echo ""
echo "📚 Quick start:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "   or run: ./activate_env.sh"
echo "2. Test installation: python test_installation.py"
echo "3. Run basic example: python examples/basic_example.py"
echo "4. Build TF-IDF index: rag-lab build-index --retriever tfidf"
echo "5. Run experiment: rag-lab run --retriever tfidf --k 5 --max-samples 100"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "💡 Remember to always activate the virtual environment before using RAG Lab!"
echo ""
