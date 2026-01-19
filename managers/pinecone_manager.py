import os
import hashlib
import time
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
from utils.text_processing import clean_pdf_text

class PineconeManager:
    """Manages Pinecone vector store operations"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.embeddings = None
        self.embedding_dimension = None
        
        self._configure_embeddings()
        self._ensure_index_exists()
    
    def _configure_embeddings(self):
        """Auto-configure embeddings based on existing index"""
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name in existing_indexes:
            index_info = self.pc.describe_index(self.index_name)
            index_dimension = index_info.dimension
            
            if index_dimension == 512:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    dimensions=512,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.embedding_dimension = 512
            elif index_dimension == 1536:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.embedding_dimension = 1536
            elif index_dimension == 3072:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.embedding_dimension = 3072
            else:
                raise ValueError(f"Unsupported dimension: {index_dimension}")
        else:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY
            )
            self.embedding_dimension = 1536
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
            time.sleep(1)
    
    def get_vectorstore(self):
        """Get LangChain Pinecone vectorstore"""
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
    
    def upload_pdf(self, pdf_path: str) -> str:
        """Upload and index a PDF file"""
        try:
            if not os.path.exists(pdf_path):
                return f"Error: File not found at {pdf_path}"
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Clean text
            cleaned_documents = []
            for doc in documents:
                cleaned_text = clean_pdf_text(doc.page_content)
                cleaned_doc = Document(
                    page_content=cleaned_text,
                    metadata=doc.metadata
                )
                cleaned_documents.append(cleaned_doc)
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                is_separator_regex=False
            )
            chunks = text_splitter.split_documents(cleaned_documents)
            
            # Add metadata
            file_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'source': pdf_path,
                    'file_id': file_id,
                    'total_chunks': len(chunks),
                    'upload_time': time.time()
                })
            
            # Upload to Pinecone
            PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
            
            return f"✓ Successfully uploaded {len(chunks)} chunks from {len(documents)} page(s)"
        except Exception as e:
            return f"Error uploading PDF: {str(e)}"