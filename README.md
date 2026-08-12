# PopQA-RAG-Judge

**PopQA-RAG-Judge** is a retrieval-augmented generation (RAG) evaluation framework built for research-grade question answering, vector retrieval, and LLM judge evaluation on the PopQA dataset. It provides a layered architecture across configuration, data ingestion, vector indexing, conversation history, local generation, and evaluation.

---

## Target Environment
- **Python**: 3.12+
- **Platform**: macOS (Apple Silicon / Intel) & Linux compatible
- **Local LLM Runtime**: [Ollama](https://ollama.com/) (running `qwen2.5:3b-instruct-q4_K_M`)
- **Judge Models**: Google Gemini (via `google-genai`), OpenAI, or local Ollama fallback (`qwen3:8b`)

---

## Project Structure

```
popqa-rag-judge/
├── config/
│   ├── config.yaml          # Core RAG, model, retrieval, and evaluation settings
│   ├── prompts.yaml         # Generation and query rewriting prompt templates
│   └── logging.yaml         # Logging configurations
├── data/
│   ├── test.tsv             # PopQA benchmark dataset
│   └── wikipedia_cache/     # Disk cache for downloaded Wikipedia articles
├── index/                   # Persisted FAISS vector index & metadata
├── sessions/                # Session logs, conversation turns, and evaluation reports
├── logs/                    # System logs (missing articles, API fallbacks)
├── reports/                 # Output evaluation summaries
├── src/
│   ├── cli/
│   │   └── app.py           # Rich interactive terminal application
│   ├── config/
│   │   └── settings.py      # Pydantic singleton configuration manager
│   ├── embeddings/
│   │   ├── base.py
│   │   └── embedder.py      # BGE embedding engine (BAAI/bge-small-en-v1.5)
│   ├── evaluation/
│   │   ├── judge.py         # Gemini / OpenAI / Ollama LLM Judge implementations
│   │   ├── json_validator.py# JSON schema validation for judge verdicts
│   │   ├── metrics.py       # Traditional metrics (EM, Token F1, BLEU, ROUGE, BERTScore)
│   │   └── pipeline.py      # Evaluation orchestrator (Session or Dataset mode)
│   ├── generator/
│   │   ├── base.py
│   │   └── ollama_client.py # Local Ollama LLM generator interface
│   ├── history/
│   │   └── session_manager.py # Conversation state and turn logger
│   ├── indexing/
│   │   ├── chunker.py       # Token-level sliding window chunker
│   │   └── index_builder.py # Concurrent article fetcher & FAISS index builder
│   ├── models/
│   │   └── data_models.py   # Dataclasses & Pydantic schemas (ConversationTurn, etc.)
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── query_rewriter.py# Contextual query rewriter
│   │   └── retriever.py     # Dense FAISS vector retriever
│   ├── storage/
│   │   ├── base.py
│   │   └── vector_store.py  # FAISS index storage manager
│   ├── utils/
│   │   └── logger.py        # System logging helpers
│   ├── wikipedia/
│   │   └── wiki_fetcher.py  # Wikipedia article downloader
│   └── runtime_bootstrap.py # Native runtime thread & OpenMP guard setup
├── main.py                  # Main application CLI entrypoint
├── run_pipeline.py          # Pre-flight environment verifier & launch wrapper
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Installation & Setup

### 1. Create & Activate Virtual Environment

```bash
# Clone or navigate to the repository
cd popqa-rag-judge

# Create Python 3.12 virtual environment
python3.12 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your API keys (optional for Cloud LLM Judge):
```env
# Optional: Required if using Google Gemini as the LLM Judge
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Required if using OpenAI as the LLM Judge
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Ensure Local Ollama Service is Running

Ensure Ollama is installed and running on your system:
```bash
ollama serve
```
*Note: The target LLM model (`qwen2.5:3b-instruct-q4_K_M`) will be verified or automatically downloaded during launch.*

---

## How to Run

### Option A: Recommended Launch (Pre-Flight Checks + Interactive CLI)

Run `run_pipeline.py` to automatically verify Python version, virtual environment status, dataset presence, Ollama connectivity, and API keys before starting the interactive app:

```bash
source .venv/bin/activate
python run_pipeline.py
```

### Option B: Interactive CLI Application

Launch `main.py` directly to enter the Rich-powered terminal QA interface:

```bash
source .venv/bin/activate
python main.py
```

#### Interactive Terminal Commands
Inside the `User >` prompt, you can enter any question or run system commands:

| Command | Description |
| :--- | :--- |
| `--help` | Show available interactive commands. |
| `--stats` | Display vector index size, session ID, and average retrieval/generation latencies. |
| `--show-context` | Enable preview table for top retrieved context chunks. |
| `--hide-context` | Disable context chunk preview table. |
| `--clear-history` | Clear sliding window conversation history. |
| `--evaluate` | Run the evaluation pipeline on current session or benchmark dataset. |
| `--rebuild-index` | Force rebuild the FAISS index from `data/test.tsv`. |
| `exit` / `quit` | Exit the CLI application. |

### Option C: Non-Interactive Evaluation Suite

To evaluate the system without entering the interactive CLI:

```bash
source .venv/bin/activate
python run_pipeline.py --evaluate
```

To evaluate a specific sample size (e.g. only 5 samples):

```bash
source .venv/bin/activate
python run_pipeline.py --evaluate - 5
# or: python run_pipeline.py --evaluate 5
# or: python run_pipeline.py --evaluate --sample-size 5
```

To force a rebuild of the vector index prior to evaluation:

```bash
source .venv/bin/activate
python run_pipeline.py --evaluate 5 --rebuild-index
```

Evaluation results will be displayed in the terminal and saved as a JSON report under `sessions/session_<timestamp>/evaluation.json`.

---

## RAG System Architecture & Workflow

1. **Wikipedia Data Ingestion & Chunking**:
   - `WikipediaFetcher` downloads target articles listed in `data/test.tsv` with disk caching in `data/wikipedia_cache/`.
   - `TokenChunker` splits text into 400-token chunks with 80-token overlap.

2. **Vector Embedding & Storage**:
   - `EmbeddingEngine` encodes text chunks using `BAAI/bge-small-en-v1.5`.
   - `VectorStore` builds and manages a FAISS index stored in `index/faiss.index` along with metadata in `index/metadata.pkl`.

3. **Query Rewriting & Retrieval**:
   - `QueryRewriter` uses conversation history to reformulate follow-up questions into standalone queries.
   - `DenseFAISSRetriever` retrieves the top-$k$ ($k=5$) relevant document chunks via inner-product/cosine similarity search.

4. **Answer Generation**:
   - `LocalOllamaGenerator` passes prompt templates (`config/prompts.yaml`), rewritten query, and context chunks to local Ollama (`qwen2.5:3b-instruct-q4_K_M`).

5. **Multi-Layer Evaluation Suite**:
   - **Mode Selection**: Evaluates completed session turns if $\ge 5$ turns exist; otherwise randomly samples rows from `data/test.tsv`.
   - **Retrieval Metrics**: Hit Rate@k, Recall@k, MRR.
   - **Answer Quality Metrics**: Alias Exact Match, Alias Token F1, BERTScore F1, Raw Exact Match, Raw Token F1, BLEU, ROUGE-L.
   - **LLM Judge Evaluation**: Grades output on Correctness, Groundedness, Completeness, Hallucination, Context Support, and issues a final `PASS`, `PARTIAL`, or `FAIL` verdict.

---

## Troubleshooting & Common Fixes

### 1. `bad interpreter: ... no such file or directory`
If you moved the project directory, the existing `.venv` path shebangs will break. Remove and recreate `.venv`:
```bash
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. OpenMP Runtime Collision (`OMP: Error #15`) on macOS
If PyTorch and FAISS duplicate `libomp` initialization on macOS, [src/runtime_bootstrap.py](file:///Users/srinivasa/Desktop/Projects/RAG_FINAL_2/popqa-rag-judge/src/runtime_bootstrap.py) handles environment flags (`KMP_DUPLICATE_LIB_OK=TRUE`) automatically. Ensure your entry point imports `src.runtime_bootstrap` before heavy imports.

### 3. Missing `google-genai` Module
If running evaluation raises `ModuleNotFoundError: No module named 'google'`:
```bash
pip install google-genai
```

### 4. Ollama Connection Error
If Ollama fails to connect:
- Verify Ollama is running (`ollama list` or `curl http://localhost:11434/api/tags`).
- Pull the model manually if needed: `ollama pull qwen2.5:3b-instruct-q4_K_M`.

---

## License

This project is built for research and evaluation purposes.
