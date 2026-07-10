---
title: Agentic Catalog Engine
emoji: 🎬🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

# Agentic Catalog Engine (ACE) 🎬🤖

An end-to-end AI application built to solve data friction in e-commerce workflows. ACE automates the extraction of dense technical specifications from chaotic manufacturer manuals (PDFs, DOCX, TXT), validates the outputs into structured JSON schemas, and serves the data through an interactive, semantic search chatbot.

This project demonstrates the transition from traditional manual data entry to a fully deterministic, agentic ETL workflow.

---

## 🚀 Key Features

* **Intelligent File Routing:** Automatically routes standard text documents through Unstructured.io for rapid processing, while intercepting complex, multi-column PDFs and routing them to the LlamaParse API to preserve table integrity and layout semantics.
* **Schema-Driven Extraction:** Utilizes the Instructor library with strict Pydantic models to forcefully govern LLM outputs. This guarantees that unstructured markdown is converted directly into a clean, strictly typed JSON database payload with zero conversational hallucination.
* **Semantic Search & RAG:** Embeds the structured catalog data using OpenAI's `text-embedding-3-small` and indexes it in a vector database for natural language querying.
* **Stateful Memory Management:** Implements a rolling memory buffer utilizing strict FIFO (First-In, First-Out) logic queueing to maintain bounded conversational context without exceeding token limits or entering repetitive generation loops.
* **Cloud Hardware Optimization:** Features a custom `@spaces.GPU` startup validation bypass, allowing the API-driven environment to deploy flawlessly on Hugging Face's high-availability CPU tiers without triggering ZeroGPU subscription blocks.

---

## 🛠️ Technology Stack

* **Frontend UI:** Gradio
* **Agentic Orchestration:** LangChain
* **Document Parsing:** LlamaParse (Vision), Unstructured.io
* **Structured Extraction:** Instructor, Pydantic
* **LLM & Embeddings:** OpenAI (`gpt-4o-mini` for high-throughput rate limit bypass, `text-embedding-3-small`)
* **Vector Storage:** Pinecone (Cloud) with Chroma DB (Local Fallback)
* **Deployment:** Hugging Face Spaces

---

## 🧠 System Architecture

1. **Ingest:** User uploads a camera manual via the Gradio interface.
2. **Parse:** System dynamically selects the optimal parsing engine based on MIME type and structural complexity.
3. **Extract:** LLM strictly conforms to a predefined `CameraGearCatalogEntry` schema to generate database-ready attributes.
4. **Index & Retrieve:** Data is embedded into a vector space, allowing users to chat directly with the technical manual via a LangChain conversational retrieval chain.

---

## 💻 Local Development Setup

To run this application on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/hetpatel-7/Agentic-Catalog-Engine.git](https://github.com/hetpatel-7/Agentic-Catalog-Engine.git)
cd Agentic-Catalog-Engine
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

**3. Configure Environment Variables:**
Create a .env file in the root directory and add your API keys:
```bash
OPENAI_API_KEY="your_openai_api_key"
LLAMA_CLOUD_API_KEY="your_llamaparse_api_key"
```

**4. Launch the application:**
```bash
python app.py
```

## 🌐 Live Demo
* The application is deployed live via Hugging Face Spaces.
* [Click here to interact with the live Agentic Catalog Engine](https://huggingface.co/spaces/hetpatel-7/Agentic-Catalog-Engine)
