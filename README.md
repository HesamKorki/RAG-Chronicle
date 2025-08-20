# RAG Lab - Comprehensive RAG Experiment Framework

A production-quality, configurable Python framework for reproducible RAG (Retrieval-Augmented Generation) experiments. Compare QA quality across retrievers that reflect the historical evolution of Information Retrieval.

## 🎯 Features

- **Multiple Retrieval Methods**: Boolean, TF-IDF, BM25, Dense (embeddings), and SOTA (dense + reranking)
- **Local Generation**: Qwen/Qwen2.5-1.5B-Instruct via Hugging Face transformers
- **Auto Device Detection**: MPS (Apple Silicon) if available, else CPU
- **SQuAD Dataset**: Default SQuAD v1.1, optional SQuAD v2 support
- **Comprehensive Evaluation**: Exact Match, F1, Recall@k, MRR@k, nDCG@k
- **Reproducible Experiments**: Deterministic seeds, full config persistence
- **Performance Tracking**: Retrieval/generation latency, throughput metrics
- **Artifact Management**: Indexes, predictions, metrics, and comparative reports

## 🏗️ Project Structure

```
rag_lab/
├── rag_lab/                    # Main package
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Configuration management
│   ├── data.py                 # SQuAD dataset handling
│   ├── indexing.py             # Index management
│   ├── retrievers/             # Retrieval implementations
│   │   ├── base.py
│   │   ├── boolean.py
│   │   ├── tfidf.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   └── sota.py
│   ├── generator/              # Text generation
│   │   ├── qwen.py
│   │   └── prompts.py
│   ├── eval/                   # Evaluation metrics
│   │   ├── squad_metrics.py
│   │   ├── retrieval_metrics.py
│   │   └── report.py
│   └── utils/                  # Utilities
│       ├── text.py
│       ├── io.py
│       ├── timing.py
│       └── seeds.py
├── configs/                    # Configuration files
│   ├── default.yaml
│   ├── retrievers.yaml
│   ├── dataset_squad.yaml
│   └── model_qwen2.5-1.5b.yaml
├── notebooks/                  # Jupyter notebooks
│   └── sanity_checks.ipynb
├── tests/                      # Unit tests
│   ├── test_boolean.py
│   ├── test_tfidf.py
│   ├── test_bm25.py
│   ├── test_dense.py
│   └── test_eval.py
├── pyproject.toml              # Project configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python >= 3.10
- macOS with Apple Silicon (M1/M2) for MPS support (optional)
- 16GB+ RAM recommended for MPS, 8GB+ for CPU
- **Note**: The framework defaults to CPU for better compatibility. To use MPS, manually set `device: "mps"` in your config

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or visit: https://docs.astral.sh/uv/getting-started/installation/
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAG-Chronicle
   ```

3. **Run the installation script** (recommended):
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. **Or install manually**:
   ```bash
   # Create virtual environment
   uv venv
   
   # Activate virtual environment
   source .venv/bin/activate
   
   # Install the package
   uv pip install -e .
   
   # Install additional dependencies
   uv pip install faiss-cpu
   ```

### Basic Usage

**Important**: Always activate the virtual environment first:
```bash
source .venv/bin/activate
# or use the activation script: ./activate_env.sh
```

#### 1. Build Retriever Indexes

First, build indexes for the retrievers you want to test:

```bash
# Build all indexes (recommended for comprehensive experiments)
rag-lab build-index --retriever boolean
rag-lab build-index --retriever tfidf
rag-lab build-index --retriever bm25
rag-lab build-index --retriever dense    # Requires more time (~5-10 minutes)
rag-lab build-index --retriever sota     # Slowest, includes reranking
```

#### 2. Run Single Experiments

Test individual retrievers with specific configurations:

```bash
# Quick test with 10 samples
rag-lab run --retriever tfidf --k 5 --max-samples 10

# Full evaluation with 1000 samples
rag-lab run --retriever dense --k 5 --max-samples 1000

# Test different k values
rag-lab run --retriever bm25 --k 3 --max-samples 100
rag-lab run --retriever bm25 --k 10 --max-samples 100
```

#### 3. Run Comprehensive Sweeps

Compare multiple retrievers across different k values:

```bash
# Compare traditional methods
rag-lab sweep --retrievers boolean tfidf bm25 --k 1 3 5 --max-samples 100

# Compare all methods (comprehensive)
rag-lab sweep --retrievers boolean tfidf bm25 dense sota --k 1 3 5 10 --max-samples 500

# Quick comparison (small sample)
rag-lab sweep --retrievers tfidf bm25 dense --k 5 --max-samples 50
```

#### 4. Understanding Results

Each experiment creates structured output in `runs/TIMESTAMP/`:

```bash
# View latest experiment results
ls runs/$(ls runs/ | tail -1)/

# Quick check of metrics
cat runs/$(ls runs/ | tail -1)/tfidf_k5/metrics.json

# View predictions
head runs/$(ls runs/ | tail -1)/tfidf_k5/predictions.jsonl
```

## 📊 Retrieval Methods

### 1. Boolean Retrieval
- **Implementation**: Inverted index with AND/OR/NOT operations
- **Use Case**: Exact keyword matching, structured queries
- **Pros**: Fast, interpretable, exact matches
- **Cons**: No ranking, limited semantic understanding

### 2. TF-IDF Retrieval
- **Implementation**: Term frequency-inverse document frequency with cosine similarity
- **Use Case**: Traditional information retrieval
- **Pros**: Well-established, interpretable, good baseline
- **Cons**: No semantic understanding, vocabulary mismatch

### 3. BM25 Retrieval
- **Implementation**: Okapi BM25 ranking function
- **Use Case**: Modern keyword-based retrieval
- **Pros**: Handles document length normalization, proven effectiveness
- **Cons**: Still keyword-based, no semantic understanding

### 4. Dense Retrieval
- **Implementation**: Sentence transformers + FAISS
- **Model**: BAAI/bge-small-en-v1.5
- **Use Case**: Semantic search, modern RAG systems
- **Pros**: Semantic understanding, handles paraphrasing
- **Cons**: Requires more computational resources

### 5. SOTA Retrieval
- **Implementation**: Dense retrieval + cross-encoder reranking
- **Models**: BGE + cross-encoder/ms-marco-MiniLM-L-12-v-2
- **Use Case**: High-precision retrieval
- **Pros**: Best accuracy, handles query-document interaction
- **Cons**: Slowest, highest computational cost

## 🔧 Configuration

The framework uses Pydantic for configuration management. Key configuration areas:

### Generator Configuration
```yaml
generator:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  max_new_tokens: 64
  temperature: 0.0
  top_p: 1.0
  context_token_budget: 2048
```

### Retriever Configuration
```yaml
dense:
  model_name: "BAAI/bge-small-en-v1.5"
  normalize_embeddings: true
  faiss_index_type: "IndexFlatIP"
  batch_size: 32
```

### Evaluation Configuration
```yaml
evaluation:
  metrics: ["exact_match", "f1"]
  retrieval_metrics: ["recall@k", "mrr@k", "ndcg@k"]
  k_values: [1, 3, 5, 10]
```

## 📈 Evaluation Metrics

### Answer Quality Metrics
- **Exact Match**: Exact string match between prediction and ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Handles SQuAD v2**: Proper handling of unanswerable questions

### Retrieval Quality Metrics
- **Recall@k**: Fraction of relevant documents retrieved in top-k
- **MRR@k**: Mean Reciprocal Rank of first relevant document
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **Precision@k**: Fraction of retrieved documents that are relevant

### Performance Metrics
- **Retrieval Latency**: Time to retrieve documents
- **Generation Latency**: Time to generate answers
- **Throughput**: Queries per second
- **Context Usage**: Token utilization efficiency

## 🗂️ Output Structure

Experiments are saved in timestamped directories under `runs/`:

```
runs/
├── 20250820_143022/              # Single experiment timestamp
│   ├── tfidf_k5/                 # Experiment: TF-IDF with k=5
│   │   ├── results.json          # Complete experimental results
│   │   ├── metrics.json          # Evaluation metrics only
│   │   ├── predictions.jsonl     # Question-answer predictions
│   │   └── config.json           # Configuration used
│   └── bm25_k3/                  # Experiment: BM25 with k=3
│       ├── results.json
│       ├── metrics.json
│       ├── predictions.jsonl
│       └── config.json
└── 20250820_150045/              # Sweep experiment timestamp
    └── sweep_results.csv         # Comparative results table

artifacts/
├── indexes/                      # Pre-built retriever indexes
│   └── squad/                    # SQuAD dataset indexes
│       ├── boolean/              # Boolean retriever index
│       ├── tfidf/                # TF-IDF retriever index
│       ├── bm25/                 # BM25 retriever index
│       ├── dense/                # Dense retriever index
│       └── sota/                 # SOTA retriever index
```

### File Contents

#### `results.json`
Complete experimental data including metrics, predictions, and configuration:
```json
{
  "metrics": {
    "exact_match": 0.4567,
    "f1": 0.5234,
    "recall@5": 0.6789,
    "avg_retrieval_time": 0.0234,
    "avg_generation_time": 1.2345
  },
  "predictions": [...],
  "config": {...}
}
```

#### `metrics.json`
Evaluation metrics only:
```json
{
  "exact_match": 0.4567,
  "f1": 0.5234,
  "recall@5": 0.6789,
  "mrr@5": 0.7890,
  "ndcg@5": 0.8123,
  "avg_retrieval_time": 0.0234,
  "avg_generation_time": 1.2345
}
```

#### `predictions.jsonl`
Question-answer predictions (one JSON object per line):
```json
{"qa_id": "56be4db0acb8001400a502ec", "question": "Which NFL team...", "ground_truth": "Denver Broncos", "predicted_answer": "Denver Broncos", "exact_match": 1.0, "f1": 1.0}
{"qa_id": "56be4db0acb8001400a502ed", "question": "Which NFL team...", "ground_truth": "Carolina Panthers", "predicted_answer": "Carolina Panthers", "exact_match": 1.0, "f1": 1.0}
```

#### `sweep_results.csv`
Comparative results across multiple retrievers:
```csv
retriever,k,exact_match,f1,recall@k,mrr@k,ndcg@k,avg_retrieval_time,avg_generation_time
boolean,5,0.1234,0.2345,0.3456,0.4567,0.5678,0.0012,1.2345
tfidf,5,0.3456,0.4567,0.5678,0.6789,0.7890,0.0234,1.2345
bm25,5,0.4567,0.5678,0.6789,0.7890,0.8901,0.0456,1.2345
```

## 🔬 Research Workflows

### Typical Research Workflow

#### 1. Initial Setup and Index Building
```bash
# Activate environment
source .venv/bin/activate

# Build all indexes (one-time setup)
rag-lab build-index --retriever boolean
rag-lab build-index --retriever tfidf  
rag-lab build-index --retriever bm25
rag-lab build-index --retriever dense    # Takes ~5-10 minutes
rag-lab build-index --retriever sota     # Takes ~10-15 minutes
```

#### 2. Preliminary Testing (Small Scale)
```bash
# Quick comparison with 10-50 samples
rag-lab sweep --retrievers boolean tfidf bm25 --k 5 --max-samples 50

# Check results
cat runs/$(ls runs/ | tail -1)/sweep_results.csv
```

#### 3. Comprehensive Evaluation (Full Scale)
```bash
# Full evaluation with 500-1000 samples
rag-lab sweep --retrievers boolean tfidf bm25 dense sota --k 1 3 5 10 --max-samples 1000

# This creates a complete comparison table
```

#### 4. Detailed Analysis
```bash
# Examine specific retriever performance
rag-lab run --retriever dense --k 5 --max-samples 1000

# View detailed predictions
head -10 runs/$(ls runs/ | tail -1)/dense_k5/predictions.jsonl
```

### Experimental Design Considerations

#### Sample Sizes
- **Quick testing**: 10-50 samples (~1-2 minutes)
- **Development**: 100-200 samples (~5-10 minutes)  
- **Research evaluation**: 500-1000 samples (~30-60 minutes)
- **Full SQuAD validation**: 10,570 samples (~2-3 hours)

#### K Values Selection
- **k=1**: Precision-focused, single best result
- **k=3**: Balanced, commonly used in RAG
- **k=5**: Standard evaluation, good recall/precision trade-off
- **k=10**: High recall, more context for generation

#### Retriever Comparison Strategy
1. **Historical progression**: `boolean → tfidf → bm25 → dense → sota`
2. **Performance tiers**: Group by computational complexity
3. **Use case specific**: Match retriever to domain requirements

### Performance Expectations

#### Timing Benchmarks (Apple M1, 16GB RAM)
- **Boolean**: ~0.001s retrieval, builds in seconds
- **TF-IDF**: ~0.02s retrieval, builds in ~30 seconds  
- **BM25**: ~0.1s retrieval, builds in ~1 minute
- **Dense**: ~0.5s retrieval, builds in ~5-10 minutes
- **SOTA**: ~2.0s retrieval (including reranking)

#### Quality Expectations (SQuAD validation)
- **Boolean**: EM ~15-25%, mainly for exact matches
- **TF-IDF**: EM ~35-45%, good baseline performance
- **BM25**: EM ~40-50%, improved over TF-IDF  
- **Dense**: EM ~55-65%, semantic understanding benefits
- **SOTA**: EM ~60-70%, best performance but slowest

## 🔬 Advanced Usage

### Custom Datasets
```python
from rag_lab.data import SquadDataset
from rag_lab.config import DatasetConfig

# Custom dataset configuration
config = DatasetConfig(
    name="your-custom-dataset",
    split="test"
)

dataset = SquadDataset(config)
dataset.load()
```

### Custom Retrievers
```python
from rag_lab.retrievers.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self):
        super().__init__("Custom")
    
    def index(self, corpus, **kwargs):
        # Your indexing logic
        pass
    
    def retrieve(self, query, k):
        # Your retrieval logic
        pass
```

### Custom Evaluation
```python
from rag_lab.eval.squad_metrics import evaluate_squad_answers

# Custom evaluation
metrics = evaluate_squad_answers(
    predictions=["answer1", "answer2"],
    ground_truths=[["gt1"], ["gt2"]]
)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=rag_lab --cov-report=html
```

## 📝 Development

### Code Style
The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Format code:
```bash
black rag_lab/
isort rag_lab/
```

### Adding New Retrievers
1. Create a new retriever class inheriting from `BaseRetriever`
2. Implement required methods: `index()`, `retrieve()`, `save_index()`, `load_index()`
3. Add configuration in `config.py`
4. Register in `cli.py`
5. Add tests in `tests/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SQuAD Dataset**: Rajpurkar et al. (2016)
- **Qwen Models**: Alibaba Cloud
- **BGE Embeddings**: BAAI
- **FAISS**: Facebook Research
- **Sentence Transformers**: UKP Lab

## 📚 References

1. Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text.
2. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
4. Johnson, J., et al. (2019). Billion-scale similarity search with GPUs.

## 📊 Results Analysis

### Working with Experiment Data

#### Quick Analysis Commands
```bash
# Get latest experiment timestamp
LATEST=$(ls runs/ | tail -1)

# View all experiments in latest run
ls runs/$LATEST/

# Check metrics for all experiments
for dir in runs/$LATEST/*/; do
  echo "=== $(basename $dir) ==="
  cat $dir/metrics.json | python3 -m json.tool
  echo
done

# Extract key metrics to CSV
echo "experiment,exact_match,f1,recall@5" > summary.csv
for dir in runs/$LATEST/*/; do
  name=$(basename $dir)
  python3 -c "import json; data=json.load(open('$dir/metrics.json')); print('$name,%.4f,%.4f,%.4f' % (data.get('exact_match',0), data.get('f1',0), data.get('recall@5',0)))" >> summary.csv
done
```

#### Python Analysis
```python
import json
import pandas as pd
from pathlib import Path

# Load experiment results
def load_experiment_results(experiment_dir):
    results = {}
    for exp_path in Path(experiment_dir).glob("*/"):
        if exp_path.is_dir():
            metrics_file = exp_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    results[exp_path.name] = json.load(f)
    return results

# Create comparison DataFrame
latest_run = sorted(Path("runs").glob("*"))[-1]
results = load_experiment_results(latest_run)
df = pd.DataFrame(results).T
print(df[['exact_match', 'f1', 'recall@5', 'avg_generation_time']])

# Statistical significance testing
from scipy import stats
# Compare two retrievers
tfidf_f1 = [...]  # Load F1 scores from predictions
bm25_f1 = [...]   # Load F1 scores from predictions
t_stat, p_value = stats.ttest_rel(tfidf_f1, bm25_f1)
print(f"Statistical significance: p={p_value:.4f}")
```

### Interpreting Results

#### Key Metrics Interpretation
- **Exact Match (EM)**: Strict accuracy, sensitive to minor differences
- **F1 Score**: More forgiving, accounts for partial matches
- **Recall@k**: How often the correct document is in top-k retrieved
- **MRR@k**: Average reciprocal rank of first relevant document
- **Generation Time**: End-to-end answer generation latency

#### Performance Analysis Patterns
1. **Boolean vs. TF-IDF**: Should see TF-IDF outperform due to ranking
2. **TF-IDF vs. BM25**: BM25 typically shows 5-10% improvement
3. **Keyword vs. Dense**: Dense retrievers excel with semantic queries
4. **Dense vs. SOTA**: SOTA adds reranking precision at cost of speed

#### Common Research Questions
- **Trade-offs**: Speed vs. accuracy across retriever types
- **Scaling**: Performance changes with different k values
- **Error Analysis**: Which question types benefit from each retriever
- **Context Quality**: How retrieved passage relevance affects generation

### Creating Publication Tables

#### LaTeX Table Generation
```python
def generate_latex_table(results_df):
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
    latex += "Retriever & EM & F1 & Recall@5 & Time(s) \\\\ \\hline\n"
    
    for idx, row in results_df.iterrows():
        latex += f"{idx} & {row['exact_match']:.3f} & {row['f1']:.3f} & "
        latex += f"{row['recall@5']:.3f} & {row['avg_generation_time']:.2f} \\\\ \\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\caption{RAG Performance Comparison}\n"
    latex += "\\end{table}"
    return latex
```

#### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Performance vs. Speed plot
plt.figure(figsize=(10, 6))
plt.scatter(df['avg_generation_time'], df['f1'], s=100)
for i, txt in enumerate(df.index):
    plt.annotate(txt, (df['avg_generation_time'].iloc[i], df['f1'].iloc[i]))
plt.xlabel('Average Generation Time (s)')
plt.ylabel('F1 Score')
plt.title('RAG Performance vs. Speed Trade-off')
plt.show()

# Metrics comparison heatmap
metrics_cols = ['exact_match', 'f1', 'recall@5', 'mrr@5']
plt.figure(figsize=(8, 6))
sns.heatmap(df[metrics_cols], annot=True, cmap='viridis', fmt='.3f')
plt.title('RAG Retriever Performance Heatmap')
plt.show()
```

## 🆘 Troubleshooting

### Common Issues

1. **Virtual environment not activated**: Always run `source .venv/bin/activate` before using RAG Lab
2. **uv not installed**: Install uv first: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **MPS not available**: Ensure you're on macOS with Apple Silicon and PyTorch >= 2.0
4. **FAISS installation issues**: Try `uv pip install faiss-cpu` instead of `pip install faiss`
5. **Out of memory**: Reduce batch sizes or use smaller models
6. **Slow dense retrieval**: Consider using smaller embedding models or GPU acceleration

### Performance Tips

1. **Index building**: Build indexes once and reuse them
2. **Batch processing**: Use larger batch sizes for dense retrieval
3. **Model caching**: HuggingFace models are cached automatically
4. **Memory management**: Monitor memory usage with large datasets

For more help, please open an issue on GitHub.

## 🎯 Quick Start Examples

### Example 1: Basic RAG Comparison
```bash
# Setup (one time)
source .venv/bin/activate
rag-lab build-index --retriever tfidf
rag-lab build-index --retriever bm25

# Run comparison
rag-lab sweep --retrievers tfidf bm25 --k 5 --max-samples 100

# View results
cat runs/$(ls runs/ | tail -1)/sweep_results.csv
```

### Example 2: Full Research Pipeline
```bash
# Build all indexes
for retriever in boolean tfidf bm25 dense sota; do
  rag-lab build-index --retriever $retriever
done

# Comprehensive evaluation
rag-lab sweep --retrievers boolean tfidf bm25 dense sota \
               --k 1 3 5 10 --max-samples 500

# Extract results for analysis
LATEST=$(ls runs/ | tail -1)
echo "Retriever,EM,F1,Recall@5,Time" > results.csv
for dir in runs/$LATEST/*/; do
  name=$(basename $dir)
  python3 -c "
import json
with open('$dir/metrics.json') as f:
    data = json.load(f)
print('$name,{:.4f},{:.4f},{:.4f},{:.2f}'.format(
    data.get('exact_match', 0),
    data.get('f1', 0), 
    data.get('recall@5', 0),
    data.get('avg_generation_time', 0)
))" >> results.csv
done
```

### Example 3: Performance Analysis
```bash
# Test different k values for best retriever
rag-lab run --retriever bm25 --k 1 --max-samples 100
rag-lab run --retriever bm25 --k 3 --max-samples 100  
rag-lab run --retriever bm25 --k 5 --max-samples 100
rag-lab run --retriever bm25 --k 10 --max-samples 100

# Compare retrieval quality vs. speed
LATEST=$(ls runs/ | tail -1)
for exp in runs/$LATEST/bm25_k*; do
  echo "=== $(basename $exp) ==="
  python3 -c "
import json
with open('$exp/metrics.json') as f:
    data = json.load(f)
print(f'Recall@k: {data.get(\"recall@\" + \"$(basename $exp)\".split(\"_k\")[1], 0):.4f}')
print(f'Generation Time: {data.get(\"avg_generation_time\", 0):.2f}s')
"
done
```

### Example 4: Error Analysis
```bash
# Run detailed experiment
rag-lab run --retriever dense --k 5 --max-samples 200

# Analyze predictions
LATEST=$(ls runs/ | tail -1)
python3 -c "
import json
correct = 0
total = 0
with open('runs/$LATEST/dense_k5/predictions.jsonl') as f:
    for line in f:
        data = json.loads(line)
        total += 1
        if data.get('exact_match', 0) == 1.0:
            correct += 1
        else:
            print(f'MISS: {data[\"question\"][:50]}...')
            print(f'  GT: {data[\"ground_truth\"]}')
            print(f'  Pred: {data[\"predicted_answer\"]}')
            print()
print(f'Accuracy: {correct}/{total} = {correct/total:.3f}')
"
```

## 📋 Checklist for Research

### Before Running Experiments
- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] All required indexes built
- [ ] Sufficient disk space (indexes can be ~1-2GB total)
- [ ] Time allocated (full evaluation can take 1-3 hours)

### During Experiments  
- [ ] Monitor memory usage (especially for dense retrievers)
- [ ] Check intermediate results (`ls runs/`)
- [ ] Verify experiments complete successfully

### After Experiments
- [ ] Back up results (`cp -r runs/ backup_runs/`)
- [ ] Extract metrics for analysis
- [ ] Validate results make sense (sanity checks)
- [ ] Document experimental conditions

### For Publications
- [ ] Record exact commands used
- [ ] Save configuration files used
- [ ] Report system specifications
- [ ] Include statistical significance tests
- [ ] Provide error analysis and examples