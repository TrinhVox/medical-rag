# ClinicalRAG — PubMed Research Assistant

A Retrieval-Augmented Generation (RAG) system for clinical research over PubMed medical literature, built with Claude API, ChromaDB, and cross-encoder reranking.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  Claude (Agentic Tool Use Loop) │
│  - Decides search query         │
│  - May search multiple times    │
│  - Generates cited answer       │
└──────────┬──────────────────────┘
           │ tool_use: search_documents
           ▼
┌─────────────────────────────────┐
│  Retrieval Pipeline             │
│  1. Bi-encoder (BGE-small)      │
│     → Top 20 from ChromaDB     │
│  2. Cross-encoder (ms-marco)    │
│     → Rerank to Top 5          │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  ChromaDB (Persistent)          │
│  500 PubMed abstracts           │
│  Domain: Type 2 Diabetes        │
└─────────────────────────────────┘
```

## How It Works

The system uses a two-stage retrieval pipeline. PubMed abstracts are embedded with a bi-encoder (BAAI/bge-small-en-v1.5) and stored in ChromaDB. At query time, the bi-encoder retrieves the top 20 candidate chunks by semantic similarity, then a cross-encoder (ms-marco-MiniLM-L-6-v2) rescores each query-document pair to rerank down to the top 5 most relevant results.

The generation layer uses Claude's tool use API in a manual agentic loop. Claude receives the user's question along with a `search_documents` tool definition. It autonomously decides what query to search for, receives the reranked results, and generates a cited answer referencing each source by PMID and title. If the retrieved context doesn't contain the answer, Claude says so explicitly rather than hallucinating.

The CLI interface supports multi-turn conversations with context management — key facts are extracted after each exchange and persisted in the system prompt, allowing old conversation turns to be trimmed without losing critical information.

## Features

- **Two-stage retrieval:** Bi-encoder retrieval + cross-encoder reranking for higher precision
- **Agentic tool use:** Claude decides when and how to search — no hardcoded retrieval logic
- **Source citations:** Every claim is attributed to a specific PMID and paper title
- **Multi-turn conversations:** Context management with automated fact extraction
- **Uncertainty flagging:** Claude explicitly states when retrieved evidence is insufficient
- **Streamlit UI:** Interactive chat interface with conversation history
- **CLI interface:** Full agentic loop with streaming and context management

## Quick Start

```bash
git clone https://github.com/TrinhVox/clinical-rag.git
cd clinical-rag

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 1. Ingest PubMed data (fetches 500 abstracts)
cd src
python ingestion.py

# 2. Chunk and embed into ChromaDB
python chunking.py

# 3. Run CLI chatbot
python generate.py

# 4. Or run Streamlit UI
cd ..
streamlit run app/main.py
```

## Project Structure

```
clinical-rag/
├── src/
│   ├── ingestion.py        # PubMed data fetching via Entrez API
│   ├── chunking.py         # Text chunking + ChromaDB embedding
│   ├── retrieval.py        # Bi-encoder retrieval + cross-encoder reranking
│   └── generate.py         # Agentic generation loop with Claude API
├── app/
│   └── main.py             # Streamlit chat interface
├── evaluation/
│   ├── generate_test_set.py # Auto-generate Q&A pairs from corpus
│   ├── run_eval.py          # Run pipeline on test set
│   └── score_ragas.py       # RAGAS evaluation scoring
├── data/                    # PubMed abstracts (JSON, one per article)
├── chroma_db/               # Persistent vector store
├── requirements.txt
└── README.md
```

## Design Decisions

**Why one chunk per abstract?**
PubMed abstracts are typically 200-300 words — already a natural semantic unit. Splitting them further would break the logical flow (background → methods → results → conclusions) without meaningful retrieval improvement. Titles are prepended to each chunk to enrich the embedding with topical context.

**Why two-stage retrieval instead of just top-5?**
Bi-encoders embed query and document independently — fast but imprecise. Cross-encoders process the query-document pair together — accurate but slow. Retrieving top-20 with the bi-encoder then reranking to top-5 with the cross-encoder combines the speed of the first with the accuracy of the second. 

**Why ChromaDB over pgvector or Pinecone?**
Local development with <1000 documents doesn't need a managed database or server infrastructure. ChromaDB runs embedded in the application process, persists to disk, and requires zero configuration. For a production deployment with millions of documents, pgvector or a managed vector database would be the appropriate choice.

**Why agentic tool use instead of hardcoded retrieval?**
Letting Claude decide the search query means it can reformulate vague questions, search multiple times with different terms, or decide not to search at all for follow-up questions in a conversation. This produces better retrieval than directly embedding the user's raw question.

## Limitations and Future Work

- **Corpus size:** 500 abstracts in one domain (Type 2 diabetes). Production systems would need broader coverage and incremental ingestion.
- **No streaming in Streamlit:** The web UI waits for the full response. Adding streaming would improve perceived latency.
- **Single-user:** No authentication or session isolation. Not suitable for multi-user deployment without additional infrastructure.

## Tech Stack

- **LLM:** Claude Opus 4.8 (generation), Claude Haiku (fact extraction)
- **Embeddings:** BAAI/bge-small-en-v1.5 via sentence-transformers
- **Reranking:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector Store:** ChromaDB (persistent, local)
- **Data Source:** PubMed via Biopython Entrez API
- **UI:** Streamlit
- **Evaluation:** RAGAS framework (pending)