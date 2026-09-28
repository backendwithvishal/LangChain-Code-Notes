# 🦜🔗 LangChain & Generative AI: From Scratch to Advanced RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2%20%2F%20v0.3-brightgreen.svg?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq%20Llama%203.3-orange.svg?logo=fastapi&logoColor=white)](https://groq.com/)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A practical, code-first masterclass and reference repository exploring **LangChain**, **LangChain Expression Language (LCEL)**, **Retrieval-Augmented Generation (RAG)**, and **Autonomous Agent architectures**. Built with real scripts, Jupyter notebooks, interactive Streamlit apps, and under-the-hood runnable implementations.

---

## 📌 Table of Contents

- [About This Repository](#-about-this-repository)
- [Repository Architecture](#-repository-architecture)
- [Module Breakdown](#-module-breakdown)
  - [1. Introduction to LangChain](#1-introduction-to-langchain)
  - [2. LangChain Components](#2-langchain-components)
  - [3. Models & Embeddings](#3-models--embeddings)
  - [4. Prompts & Interactive Apps](#4-prompts--interactive-apps)
  - [5. Structured Outputs](#5-structured-outputs)
  - [6. Output Parsers](#6-output-parsers)
  - [7. Chains & Workflows](#7-chains--workflows)
  - [8. Runnable Deep-Dive (From Scratch)](#8-runnable-deep-dive-from-scratch)
  - [9. LCEL Runnables (Part II)](#9-lcel-runnables-part-ii)
  - [10. RAG: Document Loaders](#10-rag-document-loaders)
  - [11. RAG: Text Splitters & Chunking](#11-rag-text-splitters--chunking)
  - [12. Retrievers & Vector Stores](#12-retrievers--vector-stores)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Virtual Environment](#installation--virtual-environment)
  - [API Keys Configuration](#api-keys-configuration)
- [Hands-on Demos](#-hands-on-demos)
  - [Streamlit Research Paper Summarizer](#1-streamlit-research-paper-summarizer)
  - [Terminal Chatbot](#2-terminal-chatbot)
  - [Conditional Sentiment Router](#3-conditional-sentiment-router)
  - [Semantic Chunking in Action](#4-semantic-chunking-in-action)
- [Key Architectural Lessons](#-key-architectural-lessons)
- [Contributing & Connect](#-contributing--connect)

---

## 📖 About This Repository

Building production AI applications requires moving beyond simple API calls. You need composability, deterministic structured schemas, efficient knowledge retrieval (RAG), and resilient pipeline logic.

This repository covers the complete journey:
- **Zero-to-Hero Foundation**: Understanding models, prompt templates, memory, and chains.
- **Under-the-Hood Mechanics**: Rebuilding the LCEL `Runnable` interface and `|` pipe operator from scratch in pure Python (`NakliLLM`) to demystify how LangChain chains work internally.
- **Provider Diversity**: Hands-on integration with **Groq** (ultra-fast Llama-3.3 70B & 8B), **Hugging Face Inference API / Endpoints**, **OpenRouter**, and local sentence-transformers embeddings.
- **RAG End-to-End**: Loading PDFs, web pages, CSVs, and document directories, testing splitting algorithms (fixed length, recursive, code-aware, and semantic), indexing into **FAISS** & **Chroma**, and querying with **MMR (Maximal Marginal Relevance)** and **MultiQuery** retrievers.
- **Interactive UI**: A functional Streamlit application for research paper summarization with customizable tones, lengths, and JSON prompt serialization.

---

## 🗂 Repository Architecture

```text
LangChain-Code-Notes/
├── .env.example                                  # Template for required API keys
├── requirements.txt                              # Consolidated Python dependencies
│
├── 1.Intrdouction/                               # LangChain fundamentals & ecosystem
│   ├── Introduction to LangChain.ipynb           # Architecture, benefits, and use cases
│   └── *.png                                     # Visual architecture diagrams
│
├── 2.Components/                                 # The 6 core building blocks
│   └── LangChain Components.ipynb                # Models, Prompts, Chains, Indexes, Memory, Agents
│
├── 3.Models/                                     # LLMs, ChatModels, and Embeddings
│   ├── 1.LLM/
│   │   └── 1.llm_demo.py                         # Legacy text completion LLM demo
│   ├── 2.ChatModels/
│   │   ├── 1_chatmodel_groq.py                   # ChatGroq with Llama 3.3 70B
│   │   ├── 2_chatmodel_openai.py                 # ChatOpenAI via Groq OpenAI-compatible API
│   │   └── 3_chatmodel_hf_api.py                 # ChatHuggingFace with Qwen 2.5
│   └── 3.EmbeddingModels/
│       ├── 1_embedding_hf_local.py               # Local Hugging Face embeddings (MiniLM)
│       └── document_similarity/
│           └── 1_document_similarity.py          # Vector math & similarity placeholder
│
├── 4.Prompts/                                    # Prompt engineering & interactive apps
│   ├── chatbot.py                                # Live interactive terminal chatbot
│   ├── prompt_generator.py                       # Serialize PromptTemplate to template.json
│   ├── prompt_ui.py                              # Streamlit research paper summarizer UI
│   ├── temperature.py                            # Understanding temperature & determinism
│   └── template.json                             # Serialized prompt configuration
│
├── 5.Output/                                     # Enforcing structured JSON responses
│   ├── pydantic_demo.py                          # Data validation with Pydantic BaseModel
│   ├── typeddict_demo.py                         # Lightweight typing schema with TypedDict
│   ├── with_structured_output_pydantic.py        # model.with_structured_output(BaseModel)
│   ├── with_structured_output_typeddict.py       # model.with_structured_output(TypedDict)
│   ├── with_structured_output_json.py            # model.with_structured_output(JSON Schema)
│   └── json_schema.json                          # Raw JSON schema definition
│
├── 6.Output-Parsers/                             # Parsing LLM strings into objects
│   ├── Pydantic_Output_Parser.py                 # PydanticOutputParser with format instructions
│   ├── String_Output_Parser.py                   # Manual multi-prompt piping without LCEL
│   ├── String_Output_Parser1.py                  # StrOutputParser within clean LCEL pipe
│   ├── json_Output_Parser.py                     # JsonOutputParser for structured key-values
│   └── structuredoutputparser.py                 # StructuredOutputParser & ResponseSchema
│
├── 7.Chains/                                     # Workflows & orchestration
│   ├── simple_chain.py                           # Prompt | Model | Parser + ASCII graph
│   ├── Sequential_chain.py                       # Multi-stage sequential pipeline
│   ├── Parallell_Chain.py                        # Executing parallel branches simultaneously
│   └── Conditional_Chain.py                      # Sentiment classification + RunnableBranch routing
│
├── 8.Runnable/                                   # Deconstructing LCEL from scratch
│   ├── Demo.ipynb                                # Custom chain simulation without LangChain
│   └── langchain.ipynb                           # Building custom NakliLLM & implementing `__or__`
│
├── 9.Runnable_Part_II/                           # LCEL production primitives
│   ├── runnable_sequence.py                      # Explicit RunnableSequence composition
│   ├── runnable_parallel.py                      # Concurrent Tweet & LinkedIn generation
│   ├── runnable_passthrough.py                   # Forwarding input data untouched
│   ├── runnable_branch.py                        # Dynamic conditional branching (IF/ELSE)
│   ├── runnable_lambda.py                        # Wrapping custom Python functions
│   └── *.pdf / *.docx                            # In-depth reference notes & lecture summaries
│
├── 10.RAG Document_Loaders/                      # Ingesting raw external knowledge
│   ├── text_loader.py                            # Loading plain text (.txt)
│   ├── pdf_loader.py                             # PyPDFLoader on complex PDF documents
│   ├── csv_loader.py                             # CSVLoader parsing tabular dataset rows
│   ├── directory_loader.py                       # DirectoryLoader with lazy loading over folders
│   ├── webbase_loader.py                         # WebBaseLoader live HTML scraping & QA
│   └── books/                                    # Sample PDF collection for batch ingestion
│
├── 11.RAG Text_Splitters/                        # Chunking strategies for vector indexing
│   ├── length_based.py                           # CharacterTextSplitter
│   ├── test_structure_based.py                   # RecursiveCharacterTextSplitter (paragraphs/sentences)
│   ├── python_code_splitting.py                  # Language-aware Python AST code splitter
│   ├── markdown_splitting.py                     # Markdown-aware structural splitter
│   └── semantic_meaning_splitting.py             # SemanticChunker using embeddings & variance
│
└── 12.Retrievers/                                # Precision retrieval & search algorithms
    ├── Vector_Store_retrievers.ipynb             # Chroma vector store index & similarity search
    ├── MMR_retrievers.ipynb                      # Maximal Marginal Relevance (diversity vs relevance)
    ├── MultiQuery_retriver.ipynb                 # Query expansion & multi-angle generation
    └── Wikipedia_retrievers.ipynb                # WikipediaRetriever live lookup
```

---

## 🔍 Module Breakdown

### 1. Introduction to LangChain
- **Key Concepts**: Why LangChain is essential for GenAI; limitations of standalone LLM APIs; ecosystem overview (LangChain Core, Community, Integrations, LangSmith, LangGraph).
- **Notebook**: [`1.Intrdouction/Introduction to LangChain.ipynb`](1.Intrdouction/Introduction%20to%20LangChain.ipynb).

### 2. LangChain Components
- **The Big 6**:
  1. **Models**: LLMs (text-in, text-out) vs. Chat Models (messages-in, message-out).
  2. **Prompts**: Parameterized, version-controlled prompt templates and few-shot examples.
  3. **Chains**: Connecting modular components into a cohesive workflow.
  4. **Indexes**: Preparing external data via document loaders, splitters, and vector stores.
  5. **Memory**: Managing conversational context across multiple turns.
  6. **Agents & Tools**: Giving models reasoning loops (ReAct) and external tools (search, calculators, APIs).
- **Notebook**: [`2.Components/LangChain Components.ipynb`](2.Components/LangChain%20Components.ipynb).

### 3. Models & Embeddings
- **Providers Explored**:
  - **Groq**: High-throughput inference running `llama-3.3-70b-versatile` via `langchain-groq`.
  - **Hugging Face**: Accessing remote models via `HuggingFaceEndpoint` and wrapping with `ChatHuggingFace`.
  - **OpenAI-Compatible**: Pointing `ChatOpenAI` to custom endpoints.
  - **Embeddings**: Running `sentence-transformers/all-MiniLM-L6-v2` locally via `HuggingFaceEmbeddings` for offline vector generation.

### 4. Prompts & Interactive Apps
- **Prompt Serialization**: Loading and saving prompt configurations to JSON using `langchain_core.prompts.loading._load_prompt`.
- **Streamlit Web Application** ([`prompt_ui.py`](4.Prompts/prompt_ui.py)): Interactive research paper summarizer where users pick a paper, choose an explanation style (*Beginner-Friendly*, *Technical*, *Code-Oriented*, *Mathematical*), set length, and generate targeted summaries.
- **Terminal Chatbot** ([`chatbot.py`](4.Prompts/chatbot.py)): Minimalistic conversational loop powered by Groq.
- **Temperature Guide** ([`temperature.py`](4.Prompts/temperature.py)): Detailed explanation of greedy vs. creative sampling strategies ($T=0.0$ to $1.0+$).

### 5. Structured Outputs
Instead of parsing messy model text with regular expressions, modern LangChain binds structured schemas directly into model calls:
- **`model.with_structured_output(Review)`**: Automatically validates LLM responses into typed **Pydantic** objects (`summery`, `sentiment`, `pros`, `cons`, `reviewer_name`).
- Also demonstrates schema binding with **TypedDict** and raw **JSON Schema**.

### 6. Output Parsers
Techniques for extracting typed data when using models or endpoints without native function-calling support:
- `StrOutputParser`: Strips raw `AIMessage` objects into clean UTF-8 strings.
- `JsonOutputParser`: Enforces valid JSON structure.
- `PydanticOutputParser`: Automatically injects schema instructions (`format_instructions`) into the prompt and parses output into Pydantic models.
- `StructuredOutputParser`: Extracts multiple fields defined by `ResponseSchema`.

### 7. Chains & Workflows
- **Simple LCEL Pipe**: `chain = prompt | model | parser` with visualization using `chain.get_graph().print_ascii()`.
- **Sequential Chains**: Cascading outputs from Step 1 into Step 2 inputs.
- **Conditional Routing (`Conditional_Chain.py`)**:
  - Step 1: LLM analyzes incoming customer feedback and outputs sentiment (`positive` or `negative`).
  - Step 2: `RunnableBranch` inspects sentiment and routes positive feedback to an appreciative response generator and negative feedback to a customer support resolution prompt.

### 8. Runnable Deep-Dive (From Scratch)
*How does LangChain actually chain objects with the pipe (`|`) operator?*
- In [`8.Runnable/langchain.ipynb`](8.Runnable/langchain.ipynb), we define a base class `Runnable` inheriting from `ABC` with an abstract `.invoke()` method.
- We build `NakliLLM` and custom chains implementing Python's magic method `__or__(self, other)`:
  ```python
  def __or__(self, other):
      return RunnableSequence(self, other)
  ```
- This demystifies LCEL, demonstrating that LangChain's syntax is built on standard Python object-oriented patterns.

### 9. LCEL Runnables (Part II)
Deep exploration of core LangChain Expression Language primitives:
- `RunnableParallel`: Run independent tasks concurrently (e.g., generate a Tweet and a LinkedIn post from one input topic simultaneously).
- `RunnablePassthrough`: Keeps original inputs available alongside intermediate chain outputs.
- `RunnableBranch`: Conditional execution branching.
- `RunnableLambda`: Transforming data or inserting custom Python logic into chains without writing boilerplate wrappers.

### 10. RAG: Document Loaders
Real-world data ingestion patterns:
- `TextLoader`: Plain text documents.
- `PyPDFLoader`: Parsing complex multi-page PDF documents.
- `CSVLoader`: Treating every row of a spreadsheet/dataset as an isolated document with column metadata.
- `DirectoryLoader`: Scanning local folder hierarchies and using `lazy_load()` for memory-efficient batch processing.
- `WebBaseLoader`: Scraping live HTML pages (e.g. documentation sites) and feeding scraped content into downstream QA chains.

### 11. RAG: Text Splitters & Chunking
Context windows are limited; smart chunking ensures semantic completeness:
- **Length-based Splitters**: `CharacterTextSplitter` splitting on raw character count.
- **Recursive Splitters**: `RecursiveCharacterTextSplitter` intelligently dividing text by double line breaks, single line breaks, and spaces to preserve coherent paragraphs.
- **Code & Markdown Splitters**: Preserving class and function definitions using AST rules for Python (`Language.PYTHON`) and headers for Markdown (`Language.MARKDOWN`).
- **Semantic Chunking**: `SemanticChunker` using embeddings to measure sentence-to-sentence cosine similarity and only placing chunk boundaries when semantic drift exceeds a statistical threshold.

### 12. Retrievers & Vector Stores
Converting embeddings into search engines:
- **Vector Stores**: Creating local vector stores with **FAISS** and **Chroma**.
- **Maximal Marginal Relevance (MMR)**: Balancing query relevance against result diversity so your RAG context contains unique viewpoints rather than 5 variations of the same paragraph.
- **MultiQuery Retriever**: Using an LLM to generate multiple formulations of a user's prompt to overcome keyword mismatch.
- **External Retrievers**: Connecting directly to `WikipediaRetriever` for live encyclopedic lookup.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12 installed on your machine.
- API keys for the model providers you wish to use (Groq is highly recommended for its free tier and lightning speed).

### Installation & Virtual Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/backendwithvishal/LangChain-Code-Notes.git
   cd LangChain-Code-Notes
   ```

2. **Create and activate a virtual environment:**
   - **On Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **On macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### API Keys Configuration

Copy the `.env.example` file to `.env` in the root (or in the specific folder you are executing):

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Fast inference via Groq (Free tier) -> https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here

# Hugging Face Access Token -> https://huggingface.co/settings/tokens
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# OpenRouter Key (Optional) -> https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-your_openrouter_key_here

# OpenAI Key (Optional) -> https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your_openai_key_here
```

---

## 💻 Hands-on Demos

### 1. Streamlit Research Paper Summarizer

Launch the interactive web UI:

```bash
streamlit run 4.Prompts/prompt_ui.py
```

*Select a milestone research paper (e.g. "Attention Is All You Need"), choose your target explanation style ("Beginner-Friendly", "Mathematical", etc.), and watch LangChain stream a tailored summary.*

### 2. Terminal Chatbot

Start a lightweight conversational loop in your console:

```bash
python 4.Prompts/chatbot.py
```

Type your prompt and press `Enter`. Type `exit` to end the session.

### 3. Conditional Sentiment Router

Run the dynamic sentiment classification and response chain:

```bash
python 7.Chains/Conditional_Chain.py
```

*Demonstrates classification with Pydantic followed by conditional routing through `RunnableBranch`.*

### 4. Semantic Chunking in Action

Test how embeddings identify topic transitions in a single text:

```bash
python "11.RAG Text_Splitters/semantic_meaning_splitting.py"
```

---

## 💡 Key Architectural Lessons

1. **Prefer `with_structured_output` over Manual Parsing**: Modern chat models (Groq, OpenAI, Anthropic) have dedicated tool/function calling fine-tuning. Using `model.with_structured_output(PydanticModel)` is significantly more reliable and error-free than parsing raw string responses with regex or manual parsers.
2. **Never Chunk Blindly**: Fixed-size character splitting splits sentences and code functions mid-word or mid-block. Always match your splitter to your data: `RecursiveCharacterTextSplitter.from_language` for code and `SemanticChunker` for rich domain prose.
3. **Use MMR for Research RAG**: Standard cosine similarity often retrieves 4 nearly identical sentences from the same page. Using `search_type="mmr"` ensures you get diverse perspectives across the document corpus.
4. **LCEL is Declarative**: Chaining with `|` gives you streaming, batching, asynchronous execution, and telemetry (LangSmith tracing) out of the box without changing business logic.

---

## 🤝 Contributing & Connect

Found a typo, have an improvement, or want to add another retriever pattern? Contributions are welcome!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

- **Author**: Vishal Sanam
- **GitHub**: [@backendwithvishal](https://github.com/backendwithvishal)
- **Repository**: [LangChain-Code-Notes](https://github.com/backendwithvishal/LangChain-Code-Notes)

---

⭐ **If you find these notes helpful, please star the repository!**
