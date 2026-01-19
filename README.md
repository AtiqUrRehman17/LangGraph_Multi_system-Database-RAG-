# LangGraph RAG & Text-to-SQL Router System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, modular system that intelligently routes user queries to either a **RAG (Retrieval Augmented Generation)** pipeline for PDF document analysis or a **Text-to-SQL** pipeline for PostgreSQL database queries using LangGraph workflows.

## 🌟 Features

- **🤖 Intelligent Query Routing**: Automatically determines whether to use RAG or Database based on query intent
- **📄 PDF Document Processing**: Upload and query PDF documents with advanced text preprocessing
- **🗃️ Text-to-SQL**: Natural language to SQL query generation and execution
- **🔍 MMR Retrieval**: Maximal Marginal Relevance for diverse and relevant document retrieval
- **🧩 Modular Architecture**: Clean, maintainable, and scalable codebase
- **⚡ Fast & Efficient**: Optimized chunking and embedding strategies
- **🔒 Secure**: Environment-based configuration for sensitive credentials

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Workflow Diagram](#workflow-diagram)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🏗️ Architecture

The system uses **LangGraph** to orchestrate a state machine with three main nodes:

1. **Router Node**: Analyzes query intent and routes to appropriate pipeline
2. **RAG Node**: Handles document-based queries using Pinecone vector store
3. **Database Node**: Processes database queries with Text-to-SQL conversion

### Technology Stack

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM framework and integrations
- **OpenAI**: GPT-4o-mini for LLM and text-embedding-3-small for embeddings
- **Pinecone**: Vector database for document embeddings
- **PostgreSQL**: Relational database
- **Python 3.9+**: Core language

---

## 📊 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ROUTER NODE                                 │
│  (Analyzes query intent using LLM)                              │
│                                                                 │
│  Decision: Is this about documents or database?                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│   RAG NODE       │        │  DATABASE NODE   │
│                  │        │                  │
│ 1. Query         │        │ 1. Get DB        │
│    Pinecone      │        │    Schema        │
│                  │        │                  │
│ 2. MMR           │        │ 2. Generate      │
│    Retrieval     │        │    SQL Query     │
│    (k=8 docs)    │        │                  │
│                  │        │ 3. Execute       │
│ 3. Context       │        │    SQL           │
│    Building      │        │                  │
│                  │        │ 4. Format        │
│ 4. LLM           │        │    Results       │
│    Response      │        │                  │
│    Generation    │        │ 5. Natural       │
│                  │        │    Language      │
│                  │        │    Response      │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
         ┌──────────────────────────┐
         │    FINAL RESPONSE        │
         │  (Returned to User)      │
         └──────────────────────────┘
```

### Data Flow

```
PDF Upload Flow:
PDF File → PyPDFLoader → Text Cleaning → Chunking → 
Embeddings → Pinecone Storage

Query Flow (RAG):
User Query → Router → RAG Node → Pinecone Search (MMR) → 
Context Assembly → LLM → Response

Query Flow (Database):
User Query → Router → Database Node → Schema Retrieval → 
SQL Generation → SQL Execution → Result Formatting → 
LLM → Response
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL database (for SQL queries)
- OpenAI API key
- Pinecone API key

### Step 1: Clone or Create Project Structure

```bash
mkdir langgraph_rag_sql_system
cd langgraph_rag_sql_system
```

Create the following directory structure:

```
langgraph_rag_sql_system/
├── main.py
├── requirements.txt
├── .env
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── state.py
│   └── graph.py
├── managers/
│   ├── __init__.py
│   ├── pinecone_manager.py
│   └── database_manager.py
├── nodes/
│   ├── __init__.py
│   ├── router_node.py
│   ├── rag_node.py
│   └── database_node.py
└── utils/
    ├── __init__.py
    └── text_processing.py
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
langgraph>=0.0.40
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20
langchain-pinecone>=0.0.1
python-dotenv>=1.0.0
pinecone-client>=3.0.0
psycopg2-binary>=2.9.9
pypdf>=3.17.0
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=langgraph-index

# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `PINECONE_API_KEY` | Pinecone API key | - | ✅ |
| `PINECONE_ENVIRONMENT` | Pinecone region | `us-east-1` | ❌ |
| `PINECONE_INDEX_NAME` | Pinecone index name | `langgraph-index` | ❌ |
| `DB_HOST` | PostgreSQL host | `localhost` | ✅ |
| `DB_PORT` | PostgreSQL port | `5432` | ❌ |
| `DB_NAME` | Database name | - | ✅ |
| `DB_USER` | Database user | - | ✅ |
| `DB_PASSWORD` | Database password | - | ✅ |

### Application Settings

Edit `config/settings.py` to customize:

```python
# LLM Settings
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# RAG Settings
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
MMR_K = 8                  # Number of documents to retrieve
MMR_FETCH_K = 20           # Initial fetch for MMR
MMR_LAMBDA = 0.5           # Balance relevance vs diversity
```

---

## 📁 Project Structure

```
langgraph_rag_sql_system/
│
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (gitignored)
├── README.md                   # This file
│
├── config/                     # Configuration module
│   ├── __init__.py
│   └── settings.py            # Centralized settings
│
├── core/                       # Core workflow logic
│   ├── __init__.py
│   ├── state.py               # GraphState definition
│   └── graph.py               # LangGraph workflow
│
├── managers/                   # External service managers
│   ├── __init__.py
│   ├── pinecone_manager.py    # Pinecone operations
│   └── database_manager.py    # PostgreSQL operations
│
├── nodes/                      # Workflow nodes
│   ├── __init__.py
│   ├── router_node.py         # Query routing logic
│   ├── rag_node.py            # RAG processing
│   └── database_node.py       # SQL processing
│
└── utils/                      # Utility functions
    ├── __init__.py
    └── text_processing.py     # Text preprocessing
```

---

## 💻 Usage

### Running the Application

```bash
python main.py
```

### Interactive Commands

| Command | Description | Example |
|---------|-------------|---------|
| `upload` | Upload a PDF file | `upload` → Enter path |
| `exit` | Quit the application | `exit` |
| Any text | Query the system | `what is the CGPA?` |

### Example Session

```
LangGraph RAG & Text-to-SQL Router System
==========================================================

Commands:
  'exit'   - Quit
  'upload' - Upload PDF
==========================================================

Query: upload
PDF path: /path/to/resume.pdf

✓ Successfully uploaded 15 chunks from 1 page(s)

Query: what CGPA is mentioned in the file

The CGPA mentioned is 3.23.

Query: list all students from the database

The database contains 10 students:
1. John Doe
2. Jane Smith
3. Robert Johnson
... (and 7 more)

Query: exit

Goodbye!
```

---

## 📖 API Reference

### RAGDatabaseApp Class

Main application class that orchestrates the workflow.

#### Methods

##### `__init__()`
Initializes the application with graph, Pinecone manager, and database manager.

```python
app = RAGDatabaseApp()
```

##### `upload_pdf(pdf_path: str) -> str`
Uploads and indexes a PDF file in Pinecone.

**Parameters:**
- `pdf_path` (str): Path to the PDF file

**Returns:**
- Success/error message

**Example:**
```python
result = app.upload_pdf("/path/to/document.pdf")
print(result)  # "✓ Successfully uploaded 10 chunks from 1 page(s)"
```

##### `query(user_query: str) -> dict`
Processes a user query through the LangGraph workflow.

**Parameters:**
- `user_query` (str): The user's question

**Returns:**
- Dictionary containing:
  - `query`: Original query
  - `route_decision`: "rag" or "database"
  - `final_response`: Generated response
  - `error`: Error message (if any)
  - Additional fields based on route

**Example:**
```python
result = app.query("What projects are mentioned in the document?")
print(result['final_response'])
```

---

## 🔧 Advanced Configuration

### Customizing Chunk Size

For better retrieval precision, adjust chunk parameters in `config/settings.py`:

```python
# Smaller chunks = more precise but may lose context
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Larger chunks = more context but less precise
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
```

### Customizing MMR Retrieval

Adjust MMR parameters for different retrieval strategies:

```python
MMR_K = 8              # More results = more comprehensive
MMR_FETCH_K = 20       # Higher = better candidate pool
MMR_LAMBDA = 0.5       # 0.0 = max diversity, 1.0 = max relevance
```

### Using Different Embedding Models

In `managers/pinecone_manager.py`, change the embedding model:

```python
# For higher quality (3072 dimensions)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=settings.OPENAI_API_KEY
)

# For lower cost (512 dimensions)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    openai_api_key=settings.OPENAI_API_KEY
)
```

### Using Different LLM Models

In `config/settings.py`:

```python
# For better performance
LLM_MODEL = "gpt-4o"

# For lower cost
LLM_MODEL = "gpt-3.5-turbo"
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. PDF Text Extraction Issues

**Problem:** PDF content has spaces between characters (e.g., "S k i l l s")

**Solution:** The system automatically cleans this in `utils/text_processing.py`. If issues persist, check the PDF encoding.

#### 2. Pinecone Dimension Mismatch

**Problem:** `Vector dimension X does not match the dimension of the index Y`

**Solution:** The system auto-detects and adapts. If creating a new index, ensure consistency:
```python
# Delete old index if needed
from pinecone import Pinecone
pc = Pinecone(api_key="your_key")
pc.delete_index("langgraph-index")
```

#### 3. Database Connection Failed

**Problem:** `Database connection error: could not connect to server`

**Solution:** 
- Verify PostgreSQL is running
- Check credentials in `.env`
- Test connection: `psql -h localhost -U your_user -d your_db`

#### 4. No Relevant Documents Found

**Problem:** RAG returns "No relevant documents found"

**Solution:**
1. Verify PDF was uploaded successfully
2. Check chunk samples during upload
3. Try broader queries
4. Adjust MMR parameters for more results

#### 5. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

---

## 🎯 Best Practices

### 1. PDF Preparation
- Use text-based PDFs (not scanned images)
- Ensure proper text extraction with PyPDF
- Clean PDFs work best (avoid complex layouts)

### 2. Query Formulation
- Be specific in queries
- Use natural language
- For RAG: Mention "document", "file", or "PDF"
- For DB: Mention "database", "table", or "records"

### 3. Chunk Size Selection
- **Technical docs**: Smaller chunks (500-700)
- **Narratives**: Larger chunks (1000-1500)
- **Mixed content**: Medium chunks (800-1000)

### 4. Security
- Never commit `.env` file
- Use environment variables for secrets
- Restrict database user permissions
- Use read-only SQL queries when possible

---

## 🔐 Security Considerations

1. **API Keys**: Store in `.env`, never in code
2. **Database Access**: Use least-privilege principles
3. **SQL Injection**: System validates SQL queries
4. **File Upload**: Validate PDF files before processing
5. **.env in .gitignore**: Ensure `.env` is gitignored

**.gitignore example:**
```
.env
__pycache__/
*.pyc
.venv/
venv/
```

---

## 🚀 Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t langgraph-rag-sql .
docker run -it --env-file .env langgraph-rag-sql
```

### Production Considerations

1. **Error Handling**: Add comprehensive logging
2. **Rate Limiting**: Implement API rate limits
3. **Caching**: Cache frequent queries
4. **Monitoring**: Add application monitoring
5. **Backup**: Regular database and Pinecone backups

---

## 📈 Performance Optimization

### 1. Embedding Caching
Cache embeddings to reduce API calls:
```python
# In production, implement caching layer
@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embeddings.embed_query(text)
```

### 2. Database Connection Pooling
Use connection pooling for better performance:
```python
from psycopg2 import pool
connection_pool = pool.SimpleConnectionPool(1, 20, **connection_params)
```

### 3. Async Processing
For high-volume workloads, consider async:
```python
import asyncio
from langchain.callbacks import AsyncCallbackHandler
```

---

## 🧪 Testing

### Unit Tests Example

```python
# tests/test_router.py
import pytest
from nodes.router_node import router_node
from core.state import GraphState

def test_router_rag():
    state = GraphState(
        query="What is in the document?",
        route_decision="",
        # ... other fields
    )
    result = router_node(state)
    assert result["route_decision"] == "rag"

def test_router_database():
    state = GraphState(
        query="Show all users from database",
        route_decision="",
        # ... other fields
    )
    result = router_node(state)
    assert result["route_decision"] == "database"
```

Run tests:
```bash
pytest tests/
```

---


### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Write unit tests

---

## 📝 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Email: atiqurrehmandatascientist@gmail.com

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Pinecone](https://www.pinecone.io/) for vector database
- [OpenAI](https://openai.com/) for LLM and embeddings
- [PostgreSQL](https://www.postgresql.org/) for database

---

## 📊 Project Stats

- **Language**: Python 3.9+
- **Framework**: LangGraph + LangChain
- **Lines of Code**: ~800
- **Modules**: 12
- **Dependencies**: 9

---

## 🗺️ Roadmap

- [ ] Add support for multiple PDF uploads
- [ ] Implement conversation history
- [ ] Add web interface (Streamlit/Gradio)
- [ ] Support for more document formats (DOCX, TXT)
- [ ] Multi-language support
- [ ] Query caching and optimization
- [ ] REST API endpoint
- [ ] Batch processing capabilities

---

**Made with ❤️ using LangGraph**

graph TD
    Start([User Query]) --> Router[Router Node<br/>Analyze Intent]
    
    Router -->|"Document Query<br/>(PDF, file, document)"| RAG[RAG Node]
    Router -->|"Database Query<br/>(table, records, data)"| DB[Database Node]
    
    RAG --> RAG1[Query Pinecone<br/>Vector Store]
    RAG1 --> RAG2[MMR Retrieval<br/>k=8, fetch_k=20]
    RAG2 --> RAG3[Build Context<br/>from Retrieved Chunks]
    RAG3 --> RAG4[Generate Response<br/>using LLM]
    RAG4 --> End([Return Response])
    
    DB --> DB1[Get Database<br/>Schema Info]
    DB1 --> DB2[Generate SQL Query<br/>using LLM]
    DB2 --> DB3[Execute SQL<br/>on PostgreSQL]
    DB3 --> DB4[Format Results]
    DB4 --> DB5[Generate Natural<br/>Language Response]
    DB5 --> End
    
    Upload([PDF Upload]) --> UP1[Load PDF<br/>PyPDFLoader]
    UP1 --> UP2[Clean Text<br/>Remove Spaces]
    UP2 --> UP3[Split into Chunks<br/>size=1000, overlap=200]
    UP3 --> UP4[Generate Embeddings<br/>OpenAI]
    UP4 --> UP5[Store in Pinecone<br/>Vector Database]
    UP5 --> UpEnd([Upload Complete])
    
    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style UpEnd fill:#c8e6c9
    style Router fill:#fff9c4
    style RAG fill:#f3e5f5
    style DB fill:#e1bee7
    style Upload fill:#ffe0b2