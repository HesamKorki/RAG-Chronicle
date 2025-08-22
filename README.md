# RAG Chronicle

A research framework for comparing Retrieval-Augmented Generation (RAG) systems across the historical evolution of Information Retrieval methods.

## Architecture

**Core Components:**
- **Retrievers**: Boolean, TF-IDF, BM25, Dense (BGE embeddings + FAISS), SOTA (Dense + Cross-Encoder reranking)
- **Generator**: Qwen/Qwen2.5-1.5B-Instruct with optimized prompting for concise answers
- **Dataset**: SQuAD v1.1 with automatic corpus deduplication and ground-truth mapping
- **Evaluation**: SQuAD metrics (EM, F1) + IR metrics (Recall@k, MRR@k, nDCG@k)

**Framework Design:**
```
rag_lab/
├── retrievers/          # Retrieval implementations
│   ├── base.py         # Abstract base class with Protocol
│   ├── boolean.py      # Inverted index with OR logic, stopword filtering
│   ├── tfidf.py        # Scikit-learn TfidfVectorizer + cosine similarity
│   ├── bm25.py         # rank-bm25 with Okapi BM25 scoring
│   ├── dense.py        # SentenceTransformers + FAISS IndexFlatIP
│   └── sota.py         # Dense + cross-encoder reranking
├── generator/
│   ├── qwen.py         # HuggingFace transformers with MPS/CPU support
│   └── prompts.py      # Optimized prompts for 1-5 word answers
├── eval/
│   ├── squad_metrics.py    # EM, F1 with normalization
│   └── retrieval_metrics.py # Recall, MRR, nDCG, Precision
├── config.py           # Pydantic models + YAML configuration
├── data.py             # SQuAD loading with deduplication
└── cli.py              # Experiment orchestration
```

## Retrieval Methods

### 1. Boolean Retrieval
- **Algorithm**: Inverted index with term-document mapping
- **Query Processing**: OR logic across query terms, stopword filtering
- **Scoring**: Binary relevance (1.0 for matches) with minimum term threshold
- **Complexity**: O(|query|) lookup time

### 2. TF-IDF Retrieval  
- **Algorithm**: Term Frequency × Inverse Document Frequency
- **Vectorization**: Scikit-learn TfidfVectorizer with L2 normalization
- **Similarity**: Cosine similarity between query and document vectors
- **Features**: 1-2 gram support, min_df=2 for noise reduction

### 3. BM25 Retrieval
- **Algorithm**: Okapi BM25 with length normalization
- **Parameters**: k1=1.5 (term frequency saturation), b=0.75 (length normalization)
- **Implementation**: rank-bm25 library with tokenized corpus
- **Advantages**: Document length bias correction over TF-IDF

### 4. Dense Retrieval
- **Model**: BAAI/bge-small-en-v1.5 (384-dim embeddings)
- **Index**: FAISS IndexFlatIP with normalized embeddings for cosine similarity
- **Encoding**: Batch processing (32 docs/batch) with max_seq_length=512
- **Similarity**: Inner product on normalized vectors = cosine similarity

### 5. SOTA Retrieval
- **Stage 1**: Dense retrieval (BGE) to get top-50 candidates  
- **Stage 2**: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- **Architecture**: Bi-encoder → Cross-encoder pipeline
- **Trade-off**: Highest accuracy at 10x computational cost

## Generation System

**Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Optimization**: 16 max_new_tokens (covers 100% of SQuAD answers)
- **Context**: 512 token budget with passage truncation
- **Prompting**: System prompt encourages 1-5 word answers with passages
- **Device**: CPU with float16 for speed (MPS support available)

## Evaluation Framework

**Answer Quality (SQuAD Metrics):**
- **Exact Match**: Normalized string matching with punctuation/article removal
- **F1 Score**: Token-level precision/recall harmonic mean
- **Normalization**: Lowercase, punctuation removal, article handling

**Retrieval Quality (IR Metrics):**
- **Recall@k**: Fraction of relevant docs in top-k results
- **MRR@k**: Mean reciprocal rank of first relevant document  
- **nDCG@k**: Normalized discounted cumulative gain with position weighting
- **Ground Truth**: Document ID mapping from SQuAD context to deduplicated corpus

**Performance Metrics:**
- **Retrieval Latency**: Per-query retrieval time with tqdm progress tracking
- **Generation Latency**: Per-answer generation time including tokenization
- **Throughput**: Questions per second end-to-end

## Configuration System

**Architecture**: Pydantic models with YAML defaults
- **Config Loading**: `Config.from_yaml()` with fallback to defaults
- **Validation**: Type checking and constraint validation via Pydantic
- **Modularity**: Separate configs per retriever type with inheritance
- **Reproducibility**: Fixed seeds for deterministic experiments

**Key Configurations:**
```yaml
generator:
  max_new_tokens: 16        # Optimized for SQuAD answer lengths
  context_token_budget: 512 # Speed vs context trade-off
  torch_dtype: "float16"    # Memory and speed optimization

dense:
  model_name: "BAAI/bge-small-en-v1.5"
  faiss_index_type: "IndexFlatIP"
  normalize_embeddings: true

boolean:
  min_term_threshold: 3     # Minimum matching terms for relevance
```

## Experiment Orchestration

**CLI Interface**: Three main commands
- `build-index`: Parallel index construction with progress tracking
- `run`: Single retriever experiments with configurable sampling
- `sweep`: Multi-retriever comparison with consistent QA pairs

**Data Consistency**: 
- Same QA samples across all retrievers via deterministic shuffling
- Corpus deduplication with ID mapping for fair retrieval evaluation
- Ground truth document tracking for accurate IR metrics

**Output Structure**:
```
runs/TIMESTAMP/
├── {retriever}_k{k}/
│   ├── results.json      # Complete experimental data
│   ├── metrics.json      # Evaluation metrics only  
│   ├── predictions.jsonl # Per-question predictions
│   └── config.json       # Run-specific configuration
└── sweep_results.csv     # Comparative metrics table
```

## Usage

**Installation**: `./install.sh` (uv-based dependency management)

**Basic Workflow**:
```bash
source .venv/bin/activate
rag-lab build-index        # Build all indexes
rag-lab sweep              # Run all retrievers with default settings
```

**For detailed usage**: `rag-lab --help`

## Research Applications

This framework enables systematic comparison of IR evolution in RAG systems:
- **Historical Analysis**: Boolean → TF-IDF → BM25 → Dense → SOTA progression
- **Trade-off Studies**: Accuracy vs speed across retrieval paradigms  
- **Ablation Studies**: Impact of reranking, embedding models, context size
- **Reproducible Benchmarking**: Consistent evaluation across retrieval methods

**Typical Results** (SQuAD validation):
- Boolean: ~15% EM, ~0.03s retrieval
- BM25: ~37% EM, ~0.004s retrieval  
- Dense: ~34% EM, ~0.07s retrieval
- SOTA: ~40% EM, ~1.2s retrieval

Built for research reproducibility with comprehensive logging, deterministic experiments, and structured output for statistical analysis.