import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langgraph-index")
    
    # PostgreSQL
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    # LLM Settings
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MMR_K = 8
    MMR_FETCH_K = 20
    MMR_LAMBDA = 0.5

settings = Settings()