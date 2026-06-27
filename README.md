# Enterprise Knowledge Assistant (EKA)

[![Python Version](https://img.shields.io/badge/python-3.11.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138%2B-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58%2B-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Local--Storage-red.svg?style=flat&logo=Qdrant)](https://qdrant.tech/)
[![Orchestration](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Package Manager](https://img.shields.io/badge/package--manager-uv-purple.svg)](https://github.com/astral-sh/uv)

The **Enterprise Knowledge Assistant (EKA)** is a production-grade, agentic Retrieval-Augmented Generation (RAG) system designed to deliver precise, grounded, and verifiable answers from complex enterprise documents (PDF, TXT, MD). Built on top of a highly optimized hybrid retrieval pipeline and modern LLM orchestration, EKA addresses the common challenges of RAG systems, including hallucination prevention, citation accuracy, multi-modal layout parsing (including tables), and retrieval sufficiency detection.

---

## 🚀 Key Highlights

* **Hybrid Retrieval (Dense + Sparse Search)**: Combines semantic vector search (via FastEmbed) with lexical keyword matching (BM25) to capture both contextual meaning and precise terminology.
* **RRF Rank Fusion & Cross-Encoder Reranking**: Merges dense and sparse candidates using Reciprocal Rank Fusion (RRF), followed by a transformer-based Cross-Encoder reranker to surface the most relevant context.
* **Agentic Orchestration**: Uses LangGraph to implement a state-graph-based agent that dynamically routes queries between internal document search, Exa web search, or hybrid search.
* **Docling Parsing & Table Extraction**: Utilizes layout-aware PDF extraction to accurately parse structured data, tables, and document metadata.
* **Retrieval Sufficiency Detection**: Evaluates retrieval confidence against a configurable score threshold. Bypasses generation and returns a clean refusal when no relevant evidence exists, preventing stale contexts and hallucinations.
* **Claim-Based RAG Evaluation**: Runs post-generation verification by breaking answers into atomic claims and checking them against retrieved contexts for Groundedness, Query Coverage, and Hallucination Risk.
* **Deduplicated Page-Merging Citations**: Groups citations by document and page number, automatically merging overlapping chunks and keeping only high-confidence sources.
* **Modern Streamlit Interface**: Offers a clean dashboard for multi-turn chats, document uploads, and real-time visualization of evaluation metrics.

---

## 📐 Architecture Overview

The diagram below details the data flow and execution path of a user query through the EKA pipeline:

```text
                               ┌──────────────────┐
                               │   Streamlit UI   │
                               └────────┬─────────┘
                                        │ (JSON POST Request)
                                        ▼
                               ┌──────────────────┐
                               │ FastAPI Backend  │
                               └────────┬─────────┘
                                        │ (Invoke RAG)
                                        ▼
                       ┌──────────────────────────────────┐
                       │            RAGChain              │
                       └────────┬─────────────────┬───────┘
                                │                 │
                 (Classic Mode) │                 │ (Agent Mode)
                                ▼                 ▼
                      ┌───────────────┐   ┌───────────────┐
                      │HybridRetriever│   │  LangGraph    │
                      └──────┬────────┘   │  Orchestration│
                             │            └──────┬────────┘
                             ▼                   │ (Routes to Doc/Web/Hybrid)
                (Dense/Sparse + RRF +            ▼
                 Cross-Encoder Rerank)    ┌───────────────┐
                             │            │  Search Nodes │
                             ▼            └──────┬────────┘
                      ┌───────────────┐          │
                      │  Sufficiency  │◄─────────┘
                      │  Score Check  │
                      └──────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼ (Sufficient)                    ▼ (Insufficient)
    ┌───────────────┐                 ┌───────────────┐
    │Context Builder│                 │Bypass Generat.│
    └──────┬────────┘                 │& Evaluation   │
           ▼                          └──────┬────────┘
    ┌───────────────┐                        │
    │LLM Generator  │                        │
    └──────┬────────┘                        │
           ▼                                 │
    ┌───────────────┐                        │
    │Claim Evaluator│                        │
    └──────┬────────┘                        │
           │                                 │
           ▼                                 ▼
    ┌─────────────────────────────────────────────────┐
    │     FastAPI Endpoint (chat.py) Citation Mapping  │
    └───────────────────────┬─────────────────────────┘
                            │ (JSON Response)
                            ▼
                     ┌───────────────┐
                     │ Streamlit UI  │
                     └───────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Modern, typed Python application development. |
| **Package Manager** | `uv` (by Astral) | Ultra-fast Python package resolver and environment manager. |
| **Backend Framework**| FastAPI | High-performance asynchronous REST API. |
| **Frontend Framework**| Streamlit | Clean, interactive web dashboard for chat and file upload. |
| **Orchestration** | LangChain & LangGraph | State-graph-based agentic workflows and pipeline orchestration. |
| **Vector Database** | Qdrant (Local) | Hybrid vector storage and BM25 lexical index provider. |
| **Dense Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`) | Fast local embedding generator (384-dimensional). |
| **Sparse Retrieval** | BM25 (`Qdrant/bm25`) | Lexical token matching. |
| **Reranking Model** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Transformer-based cross-encoder for precise semantic relevance. |
| **LLM Provider** | Mistral AI (Default) / Gemini / Groq | Configurable enterprise LLM endpoints via LangChain. |
| **Parser / Ingestion** | Docling | Layout-aware document parsing, table extraction, and metadata enrichment. |

---

## 📂 Folder Structure

```text
├── backend/               # FastAPI asynchronous backend application
│   ├── agents/            # LangGraph agent definitions, state schemas, and tools
│   │   ├── providers/     # Direct web search provider service wrappers (e.g., Exa API)
│   │   ├── graph.py       # State-graph transition compiled logic
│   │   ├── nodes.py       # State machine node functions (routers, search nodes)
│   │   ├── state.py       # Graph session state definitions
│   │   ├── tools.py       # Modular RAG search tools exposed to agents
│   │   └── web_search.py  # Web search orchestration service
│   ├── api/               # API endpoint route controllers
│   │   ├── chat.py        # /chat endpoint mapping query evaluations and citations
│   │   ├── documents.py   # /documents endpoint handling document catalogue operations
│   │   ├── ingest.py      # /ingest endpoint for uploading and vectorizing new files
│   │   └── search.py      # /search endpoint executing retrieval diagnostics
│   ├── config/            # Centralized settings and environment variable parsing
│   │   └── settings.py    # Central Pydantic BaseSettings class
│   ├── core/              # Core infrastructure utilities
│   │   └── logging.py     # System logging configurations
│   ├── embeddings/        # Embedding models for vector and lexical databases
│   │   ├── dense_embedder.py  # FastEmbed client generating dense vectors
│   │   ├── sparse_embedder.py # Lexical tokenizer generating sparse metrics
│   │   └── hybride_embedder.py # Hybrid utility mapping combined inputs
│   ├── ingestion/         # Layout-aware text parsing and database indexing pipelines
│   │   ├── chunker.py     # Text chunking and overlap strategizer
│   │   ├── metadata.py    # Document metadata extraction and enrichment
│   │   ├── parser.py      # Docling file extractor
│   │   ├── table_parser.py # Docling-based table parser converting tables to Markdown
│   │   └── pipeline.py    # Main ingestion worker pushing chunks to Qdrant
│   ├── llm/               # LLM integration, prompt templates, and evaluation layers
│   │   ├── evaluation.py  # Evaluation manager calculating grounding/coverage/hallucination
│   │   ├── evaluator.py   # AnswerEvaluator orchestration interface
│   │   ├── generator.py   # Answer generation and context optimization manager
│   │   ├── model.py       # LLM provider factory (Gemini, Groq, Mistral)
│   │   ├── prompts.py     # Prompt templates for generation and metrics
│   │   └── rag_chain.py   # RAGChain pipeline execution orchestrator
│   ├── retrieval/         # Multi-stage hybrid search and reranking execution
│   │   ├── dense.py       # Dense vector distance calculations against Qdrant
│   │   ├── hybrid.py      # Lexical + Dense search execution with RRF scores
│   │   ├── reranker.py    # Cross-Encoder model score evaluation
│   │   └── retriever.py   # HybridRetriever entrypoint and sufficiency validator
│   ├── schemas/           # Pydantic data validation classes
│   │   ├── chat.py        # Request and response models for chat interactions
│   │   ├── documents.py   # Document list and deletion response schemas
│   │   ├── ingest.py      # File upload status response models
│   │   └── search.py      # Diagnostic retrieval schemas
│   ├── services/          # Supporting domain business logic
│   │   ├── citation_service.py # Citation formatting and page-level deduplication
│   │   └── document_service.py # Document indexing and registry transactions
│   ├── vectorstore/       # Local database connectivity
│   │   └── qdrant.py      # Local Qdrant collection initializer and query client
│   └── main.py            # FastAPI main router entrypoint
├── frontend/              # Interactive user dashboard built with Streamlit
│   ├── assests/           # UI media, logos, and custom layout assets
│   ├── components/        # Sidebar filters and chat messages visual layouts
│   ├── pages/             # App page modules (Chat interface and Ingestion dashboard)
│   ├── services/          # API communication clients and local state managers
│   ├── app.py             # Streamlit app routing and entrypoint navigation
│   └── settings.py        # Streamlit frontend properties
├── .gitignore             # Standard git ignore list for temp and log folders
├── pyproject.toml         # Python packaging dependencies configured via uv
├── uv.lock                # Locked package dependency versions
└── README.md              # Project documentation
```

---

## ✨ Features Detail

| Feature | Technical Implementation |
| :--- | :--- |
| **Hybrid Retrieval** | Combines Dense vector embeddings (BGE-small) and lexical BM25 scores. |
| **RRF Fusion** | Merges Dense and Sparse rank positions using Reciprocal Rank Fusion (RRF) to leverage both semantic and exact-match capabilities. |
| **Cross-Encoder Reranking** | Re-scores candidates using a MS-MARCO Cross-Encoder model to bubble up the most contextually relevant chunks. |
| **LangGraph Agent Mode** | Executes an agentic state graph that routes between document search, Exa web search, and hybrid search based on query type. |
| **Table Extraction** | Docling parses tables from complex enterprise PDFs and represents them cleanly as Markdown for accurate LLM extraction. |
| **Retrieval Sufficiency** | Validates top reranker score against a minimum threshold (`0.35`). Triggers immediate clean refusal if evidence is insufficient. |
| **Claim-Based Evaluation** | Extracts atomic claims from the generated answer and verifies them against context using LLM reasoning (Grounding, Coverage, Hallucination Risk). |
| **Page-Merging Citations** | Automatically merges citations originating from the same page, keeping the chunk with the highest similarity score. |
| **Document Management** | Exposes REST APIs to list, upload, parse, and delete documents with real-time Qdrant index updates. |

---

## 🔍 Retrieval Pipeline

EKA implements a state-of-the-art multi-stage retrieval pipeline:

```text
Query ──┬──> Dense Search (FastEmbed BGE-small) ──> Top 20 Candidates ──┐
        └──> Sparse Search (BM25 Lexical) ───────> Top 20 Candidates ──┼──> RRF Fusion (Top 50 Chunks) ──> Cross-Encoder Reranking (Top 5 Chunks) ──> Context Builder
```

1. **Dense Retrieval**: Generates vector embeddings for the query and extracts the top 20 nearest chunks from the Qdrant dense vector index using cosine similarity.
2. **Sparse Retrieval**: Executes keyword-based token matching (BM25) on the Qdrant document collection to capture precise terminology and numbers, returning the top 20 candidates.
3. **Reciprocal Rank Fusion (RRF)**: Merges rank positions from the dense and sparse candidate pools. Candidates are scored based on:
   $$\text{RRF Score}(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$
   This bubbles up chunks that score consistently well in both semantic and token matching.
4. **Cross-Encoder Reranking**: The top 50 candidates are fed into `ms-marco-MiniLM-L-6-v2`. This model evaluates the query-chunk pairs simultaneously, producing highly accurate relevance scores.
5. **Context Building**: Chunks scoring above the retrieval threshold are passed to the Context Builder, which resolves duplicates, merges adjacent chunks from the same page, and structures the text and tables under budget limits.
6. **Answer Generation**: The structured context and query are passed to the configured LLM (e.g., Mistral AI) to generate a concise, grounded answer.
7. **Claim-Based Evaluation**: The generated answer is validated post-hoc to verify factual correctness and calculate precision metrics.

---

## 🤖 Agentic Workflow

When **Agent Mode** is enabled, EKA executes a stateful graph built using **LangGraph**:

```text
               ┌───────────────┐
               │  Query Router │
               └───────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 ┌────────────┐ ┌────────────┐ ┌────────────┐
 │ Documents  │ │ Web Search │ │   Hybrid   │
 │   Search   │ │ (Exa API)  │ │   Search   │
 └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
            ┌─────────────────────┐
            │  Answer Generation  │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │  Answer Evaluation  │
            └─────────────────────┘
```

* **Query Router Node**: Analyzes the query using a structured LLM call and routes execution along three paths: `documents` (internal knowledge base), `web` (live web queries), or `hybrid` (cross-checking both).
* **Document Search Node**: Queries the internal Qdrant index. Sets sufficiency variables based on relevance thresholds.
* **Web Search Node**: Executes queries against the **Exa API**, fetching relevant web content.
* **Hybrid Search Node**: Triggers parallel execution of both search paths and combines the results.
* **Answer Node**: Builds the context dynamically from active search results and generates the final answer.
* **Evaluation Node**: Passes the generated answer, context, and retrieval metrics to the claim verification engine to verify safety and accuracy.

---

## 📊 Evaluation Pipeline

EKA protects against hallucinations using a strict, multi-dimensional evaluation pipeline:

```text
Answer ──> Claim Extractor ──> Atomic Claims ──> Claim Verifier (Context + Claims) ──> Supported / Unsupported ──> Metrics Calculation
```

* **Grounding Score**: The ratio of verified atomic claims to total claims:
  $$\text{Grounding Score} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$
  If any claim is found to be unsupported, EKA tags the answer as containing hallucinated information.
* **Query Coverage Score**: Extracts the core information units required to satisfy the user query, and calculates how many of those units are addressed in the generated answer.
* **Hallucination Risk**: Assesses mismatch risks (e.g., entity or numerical substitutions) and flags risk levels as `Low`, `Medium`, or `High`.
* **Retrieval Sufficiency Detection**: A pre-emptive safeguard. If the top-scoring candidate fails to meet the `retrieval_min_score` (default: `0.35`), evaluations bypass metric calculations, set status to `INSUFFICIENT_EVIDENCE`, and return a standard refusal answer with zero sources attached.

---

## ⚙️ Installation & Setup

### Prerequisites
* Python >= 3.11
* [uv](https://github.com/astral-sh/uv) (Astral's fast Python package installer)

### 1. Clone the Repository
```bash
git clone https://github.com/kaushik238P/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant
```

### 2. Install Dependencies
Initialize the virtual environment and install all packages in one step using `uv`:
```bash
uv sync
```

### 3. Configure Environment Variables
Create a `.env` file in the root of the project:
```bash
cp .env.example .env
```
Fill in the required configuration variables:
```ini
# API Keys

MISTRAL_API_KEY="your-mistral-api-key"
EXA_API_KEY="your-exa-api-key"

# Active LLM Configuration
LLM_PROVIDER="llm_provide"
LLM_MODEL="llm_model"
LLM_TEMPERATURE=0.0

# Local vector store path
QDRANT_PATH="Database/"
```

### 4. Run the Backend
Start the FastAPI server:
```bash
uv run uvicorn backend.main:app --port 8000 --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

### 5. Run the Frontend
In a new terminal, launch the Streamlit interface:
```bash
uv run streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.

---


## 💡 Typical Usage Flow

1. **Upload Documents**: Open the **Upload** page in the Streamlit UI and upload PDF or text files. The Docling parser will structure, chunk, and index the pages.
2. **Build / Verify Index**: Monitor document listings and index states on the dashboard.
3. **Ask Questions**: Open the **Chat** interface and query the knowledge base.
4. **Interactive Evaluations**: Review inline citation links and verify evaluations (Grounding Score, Coverage, and Hallucination Risk) returned below each assistant response.
5. **Enable Agent Mode**: Toggle **Agent Mode** in the sidebar to allow dynamic routing to Exa web search for live queries.

---

## 📸 Screenshots

## 🏠 Home Page

The main chat interface for interacting with the knowledge base.

![Home Page](images/home.png)

## 📤 Document Upload

Upload and manage PDF, TXT, and Markdown documents.

![Upload](images/upload.png)

## 💬 Hybrid RAG Answer

The assistant answers questions using retrieved document context.

![RAG Answer](images/rag-answer.png)

## 📊 Evaluation Pipeline

Every answer is evaluated using Grounding, Coverage, and Hallucination metrics.

![Evaluation](images/evaluation.png)

## 🤖 Agent Mode + Web Search

Agent mode can combine document retrieval with web search.

![Agent Mode](images/agent-mode.png)

## 🤝 Acknowledgements

* [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework.
* [Streamlit](https://streamlit.io/) - Fast dashboarding app framework.
* [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful agent orchestration.
* [Qdrant](https://qdrant.tech/) - High-performance hybrid vector database.
* [Docling](https://github.com/DS4SD/docling) - Advanced document parser and layouter.
* [FastEmbed](https://github.com/qdrant/fastembed) - High-speed vector embeddings generation.
* [SentenceTransformers](https://sbert.net/) - State-of-the-art embeddings and rerankers.