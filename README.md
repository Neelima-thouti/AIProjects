# Insurance Policy Q&A Assistant (RAG-based)

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about insurance policy documents, with every answer grounded in the actual source text — no hallucinated coverage details, no outside knowledge. Includes a multi-policy comparison feature to compare clauses across two policies side by side.

## Why this project

Insurance policy documents are long, dense, and full of clauses that change meaning based on exact wording. A generic chatbot answering from general knowledge is actively dangerous here — a wrong answer about coverage or exclusions has real financial consequences. This project forces every answer to be grounded in retrieved source text, and explicitly flags when it isn't confident.

## Architecture

```
Raw PDFs → Text Extraction → Chunking → Embedding → Vector Store (FAISS)
                                                            ↓
User Question → Embedding → Similarity Search → Relevant Chunks → LLM → Answer + Sources
```

| Stage | Tool | Purpose |
|---|---|---|
| PDF extraction | pdfplumber | Pull text from policy PDFs, page by page |
| Chunking | LangChain (RecursiveCharacterTextSplitter) | Split text into 800-char overlapping chunks |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) | Convert text to vectors, runs locally, free |
| Vector store | FAISS | Fast similarity search over chunk vectors |
| LLM | Groq API (Llama 3.1 8B) | Generate grounded answers, free tier |
| UI | Streamlit | Interactive Q&A and comparison interface |

## Features

- **Grounded Q&A** — answers only from retrieved policy text, explicitly states when context is insufficient
- **Source citations** — every answer cites the document and page number it came from
- **Confidence flagging** — warns the user when the best-matching retrieved content isn't a strong match
- **Multi-policy comparison** — retrieves separately from two policies (to guarantee balanced representation) and compares them side by side on a given topic

## Setup

```bash
git clone <this-repo-url>
cd insurance-rag-assistant
pip install -r requirements.txt
```

Create a `.env` file with a free Groq API key (from console.groq.com):
```
GROQ_API_KEY=your_key_here
```

Add policy PDFs to `data/policies/`, then build the vector store:
```bash
python src/embed_store.py
```

Run the app:
```bash
streamlit run src/app.py
```

## Known limitations

- Retrieval is purely semantic (no keyword/exact-match fallback) — weak on queries needing exact figures like sum-insured amounts
- Confidence threshold is empirically set, not statistically validated
- No formal evaluation framework yet (retrieval precision/recall not systematically measured)
- Uses `IndexFlatL2` (exact search) — fine at this scale, would need a different index type or a managed vector DB at much larger scale

## Tech stack

Python, LangChain, FAISS, Sentence-Transformers, Groq API, Streamlit, pdfplumber
