"""
LangGraph RAG and Text-to-SQL Router System
A complete workflow that routes user queries to either RAG or Database based on intent
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Database imports
import psycopg2
from psycopg2.extras import RealDictCursor

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec

# PDF processing
import tempfile
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langgraph-rag-index")

# Embedding configuration - will be set dynamically based on index
EMBEDDING_MODEL = None
EMBEDDING_DIMENSION = None

# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ==================== State Definition ====================
class GraphState(TypedDict):
    """State for the LangGraph workflow"""
    query: str
    route_decision: str
    rag_response: str
    sql_query: str
    sql_result: str
    final_response: str
    error: str
    messages: Annotated[list, operator.add]


# ==================== LLM Initialization ====================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# Embeddings will be initialized after checking Pinecone index dimension
embeddings = None


# ==================== Pinecone Setup ====================
class PineconeManager:
    """Manages Pinecone vector store operations"""
    
    def __init__(self):
        global embeddings, EMBEDDING_MODEL, EMBEDDING_DIMENSION
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        
        # Detect dimension and configure embeddings
        self._configure_embeddings()
        self._ensure_index_exists()
        
    def _configure_embeddings(self):
        """Auto-configure embeddings based on existing index or create new index"""
        global embeddings, EMBEDDING_MODEL, EMBEDDING_DIMENSION
        
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name in existing_indexes:
            # Get existing index dimension
            index_info = self.pc.describe_index(self.index_name)
            index_dimension = index_info.dimension
            
            # Map dimension to appropriate model
            dimension_to_model = {
                512: "text-embedding-3-small",  # with dimensions parameter
                1536: "text-embedding-3-small",  # default
                3072: "text-embedding-3-large",
            }
            
            if index_dimension == 512:
                # Use text-embedding-3-small with reduced dimensions
                EMBEDDING_MODEL = "text-embedding-3-small"
                EMBEDDING_DIMENSION = 512
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    dimensions=512,  # Explicitly set to 512
                    openai_api_key=OPENAI_API_KEY
                )
                print(f"✓ Configured embeddings: {EMBEDDING_MODEL} with {EMBEDDING_DIMENSION} dimensions")
            elif index_dimension == 1536:
                EMBEDDING_MODEL = "text-embedding-3-small"
                EMBEDDING_DIMENSION = 1536
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    openai_api_key=OPENAI_API_KEY
                )
                print(f"✓ Configured embeddings: {EMBEDDING_MODEL} with {EMBEDDING_DIMENSION} dimensions")
            elif index_dimension == 3072:
                EMBEDDING_MODEL = "text-embedding-3-large"
                EMBEDDING_DIMENSION = 3072
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    openai_api_key=OPENAI_API_KEY
                )
                print(f"✓ Configured embeddings: {EMBEDDING_MODEL} with {EMBEDDING_DIMENSION} dimensions")
            else:
                raise ValueError(f"Unsupported index dimension: {index_dimension}. Supported: 512, 1536, 3072")
            
            print(f"✓ Using existing Pinecone index: {self.index_name} (dimension: {index_dimension})")
        else:
            # No existing index, use default configuration
            EMBEDDING_MODEL = "text-embedding-3-small"
            EMBEDDING_DIMENSION = 1536
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            print(f"✓ Configured embeddings: {EMBEDDING_MODEL} with {EMBEDDING_DIMENSION} dimensions")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            print(f"✓ Created new Pinecone index: {self.index_name}")
            # Wait for index to be ready
            import time
            time.sleep(1)
    
    def get_vectorstore(self):
        """Get LangChain Pinecone vectorstore"""
        from langchain_pinecone import PineconeVectorStore
        
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
    
    def upload_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Upload and index a PDF file"""
        try:
            # Verify file exists
            if not os.path.exists(pdf_path):
                return f"Error: File not found at {pdf_path}"
            
            print(f"\nLoading PDF from: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"✓ Loaded {len(documents)} pages from PDF")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            print(f"✓ Split into {len(chunks)} chunks")
            print(f"✓ Uploading to Pinecone index '{self.index_name}' (dimension: {EMBEDDING_DIMENSION})...")
            
            # Add to vectorstore using the new method
            from langchain_pinecone import PineconeVectorStore
            
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=self.index_name,
                pinecone_api_key=PINECONE_API_KEY
            )
            
            print(f"✓ Upload complete!\n")
            return f"✓ Successfully uploaded and indexed {len(chunks)} chunks from PDF ({len(documents)} pages)"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\nDetailed error:\n{error_details}")
            return f"Error uploading PDF: {str(e)}"


# ==================== Database Manager ====================
class DatabaseManager:
    """Manages PostgreSQL database operations"""
    
    def __init__(self):
        self.connection_params = {
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD
        }
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)
    
    def get_schema_info(self):
        """Get database schema information"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            schema_info = {}
            for (table_name,) in tables:
                # Get columns for each table
                cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """)
                columns = cursor.fetchall()
                schema_info[table_name] = columns
            
            cursor.close()
            conn.close()
            
            return schema_info
        except Exception as e:
            return {"error": str(e)}
    
    def execute_query(self, query: str):
        """Execute SQL query and return results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
            else:
                conn.commit()
                results = {"message": "Query executed successfully"}
            
            cursor.close()
            conn.close()
            
            return results
        except Exception as e:
            return {"error": str(e)}


# ==================== Node Functions ====================

def router_node(state: GraphState) -> GraphState:
    """Routes the query to either RAG or Database based on intent"""
    query = state["query"]
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant that determines whether a user query should be handled by:
1. RAG (Retrieval Augmented Generation) - for questions about uploaded PDF documents
2. DATABASE - for questions that require querying a PostgreSQL database

Analyze the user's query and respond with ONLY one word: either 'RAG' or 'DATABASE'

Examples:
- "What does the document say about..." -> RAG
- "Summarize the PDF content" -> RAG
- "Show me all users in the database" -> DATABASE
- "What are the sales figures for last month?" -> DATABASE
- "Query the customer table" -> DATABASE

Query: {query}

Route to:"""),
        ("human", "{query}")
    ])
    
    chain = router_prompt | llm | StrOutputParser()
    decision = chain.invoke({"query": query}).strip().upper()
    
    # Validate decision
    if "RAG" in decision:
        route_decision = "rag"
    elif "DATABASE" in decision or "DB" in decision:
        route_decision = "database"
    else:
        # Default to RAG if unclear
        route_decision = "rag"
    
    state["route_decision"] = route_decision
    state["messages"].append(AIMessage(content=f"Routing to: {route_decision.upper()}"))
    
    return state


def rag_node(state: GraphState) -> GraphState:
    """Handles RAG queries using Pinecone vector store"""
    query = state["query"]
    
    try:
        pinecone_manager = PineconeManager()
        vectorstore = pinecone_manager.get_vectorstore()
        
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            state["rag_response"] = "No relevant documents found in the knowledge base."
            state["final_response"] = state["rag_response"]
            return state
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response using LLM
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context from documents.
Use the context below to answer the user's question accurately. If the answer is not in the context, say so.

Context:
{context}"""),
            ("human", "{query}")
        ])
        
        chain = rag_prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})
        
        state["rag_response"] = response
        state["final_response"] = response
        state["messages"].append(AIMessage(content=f"RAG Response: {response}"))
        
    except Exception as e:
        error_msg = f"Error in RAG processing: {str(e)}"
        state["error"] = error_msg
        state["final_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
    
    return state


def database_node(state: GraphState) -> GraphState:
    """Handles database queries with Text-to-SQL"""
    query = state["query"]
    
    try:
        db_manager = DatabaseManager()
        schema_info = db_manager.get_schema_info()
        
        if "error" in schema_info:
            error_msg = f"Database connection error: {schema_info['error']}"
            state["error"] = error_msg
            state["final_response"] = error_msg
            return state
        
        # Format schema for prompt
        schema_text = ""
        for table, columns in schema_info.items():
            schema_text += f"\nTable: {table}\n"
            for col_name, col_type in columns:
                schema_text += f"  - {col_name} ({col_type})\n"
        
        # Generate SQL query
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Given the database schema below, generate a PostgreSQL query to answer the user's question.

Database Schema:
{schema}

IMPORTANT: 
- Return ONLY the SQL query, no explanations
- Use proper PostgreSQL syntax
- Ensure the query is safe and read-only when possible
- Do not use DROP, DELETE, or TRUNCATE unless explicitly requested"""),
            ("human", "{query}")
        ])
        
        chain = sql_prompt | llm | StrOutputParser()
        sql_query = chain.invoke({"schema": schema_text, "query": query}).strip()
        
        # Clean up SQL query (remove markdown formatting if present)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        state["sql_query"] = sql_query
        
        # Execute query
        results = db_manager.execute_query(sql_query)
        
        if isinstance(results, dict) and "error" in results:
            error_msg = f"SQL execution error: {results['error']}"
            state["error"] = error_msg
            state["final_response"] = f"Generated SQL:\n{sql_query}\n\nError: {error_msg}"
        else:
            # Format results
            state["sql_result"] = str(results)
            
            # Generate natural language response
            result_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that explains database query results in natural language.
Given the SQL query and its results, provide a clear and concise explanation.

SQL Query: {sql_query}

Results: {results}"""),
                ("human", "Please explain these results in a user-friendly way.")
            ])
            
            chain = result_prompt | llm | StrOutputParser()
            explanation = chain.invoke({
                "sql_query": sql_query,
                "results": str(results)
            })
            
            final_response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Results:**\n{explanation}"
            state["final_response"] = final_response
        
        state["messages"].append(AIMessage(content=state["final_response"]))
        
    except Exception as e:
        error_msg = f"Error in database processing: {str(e)}"
        state["error"] = error_msg
        state["final_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
    
    return state


def route_decision(state: GraphState) -> Literal["rag", "database"]:
    """Conditional edge function to route based on decision"""
    return state["route_decision"]


# ==================== Build Graph ====================
def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("database", database_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "database": "database"
        }
    )
    
    # Add edges to END
    workflow.add_edge("rag", END)
    workflow.add_edge("database", END)
    
    return workflow.compile()


# ==================== Main Application ====================
class RAGDatabaseApp:
    """Main application class"""
    
    def __init__(self):
        self.graph = build_graph()
        self.pinecone_manager = PineconeManager()
        self.db_manager = DatabaseManager()
    
    def upload_pdf(self, pdf_path: str):
        """Upload a PDF to the RAG system"""
        return self.pinecone_manager.upload_pdf(pdf_path)
    
    def query(self, user_query: str):
        """Process a user query through the workflow"""
        initial_state = {
            "query": user_query,
            "route_decision": "",
            "rag_response": "",
            "sql_query": "",
            "sql_result": "",
            "final_response": "",
            "error": "",
            "messages": [HumanMessage(content=user_query)]
        }
        
        result = self.graph.invoke(initial_state)
        return result


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Initialize application
    app = RAGDatabaseApp()
    
    print("=" * 80)
    print("LangGraph RAG & Text-to-SQL Router System")
    print("=" * 80)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode (type 'exit' to quit, 'upload' to upload PDF)")
    print("=" * 80)
    
    while True:
        user_input = input("\nYour query: ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'upload':
            pdf_path = input("Enter PDF path: ").strip()
            result = app.upload_pdf(pdf_path)
            print(result)
            continue
        
        if not user_input:
            continue
        
        result = app.query(user_input)
        print(f"\n{'='*60}")
        print(f"Route: {result['route_decision'].upper()}")
        print(f"{'='*60}")
        print(f"\n{result['final_response']}")