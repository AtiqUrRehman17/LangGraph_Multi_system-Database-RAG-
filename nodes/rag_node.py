from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import GraphState
from managers.pinecone_manager import PineconeManager
from config.settings import settings

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    openai_api_key=settings.OPENAI_API_KEY
)

def rag_node(state: GraphState) -> GraphState:
    """Handles RAG queries using Pinecone vector store"""
    query = state["query"]
    
    try:
        pinecone_manager = PineconeManager()
        vectorstore = pinecone_manager.get_vectorstore()
        
        # Use MMR for diverse retrieval
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=settings.MMR_K,
            fetch_k=settings.MMR_FETCH_K,
            lambda_mult=settings.MMR_LAMBDA
        )
        
        if not docs:
            response = "No relevant documents found. Please upload a PDF document first."
            state["rag_response"] = response
            state["final_response"] = response
            return state
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.

Answer the question using only the information in the context below.
If the answer is not in the context, say "I cannot find this information in the provided documents."
Be direct and concise in your response.

Context:
{context}"""),
            ("human", "{query}")
        ])
        
        chain = rag_prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})
        
        state["rag_response"] = response
        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))
        
    except Exception as e:
        error_msg = f"Error in RAG processing: {str(e)}"
        state["error"] = error_msg
        state["final_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
    
    return state