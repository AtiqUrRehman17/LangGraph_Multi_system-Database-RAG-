from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import GraphState
from config.settings import settings

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    openai_api_key=settings.OPENAI_API_KEY
)

def router_node(state: GraphState) -> GraphState:
    """Routes the query to either RAG or Database based on intent"""
    query = state["query"]
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant. Determine if the query is about:
1. RAG - Questions about uploaded PDF documents, file content
2. DATABASE - Questions requiring database queries, tables, records

Respond with ONLY one word: either 'RAG' or 'DATABASE'

Examples:
- "What does the document say..." -> RAG
- "What CGPA is in the file" -> RAG
- "List projects from file" -> RAG
- "Show all users from database" -> DATABASE
- "Query customer table" -> DATABASE

Query: {query}"""),
        ("human", "{query}")
    ])
    
    chain = router_prompt | llm | StrOutputParser()
    decision = chain.invoke({"query": query}).strip().upper()
    
    route_decision = "rag" if "RAG" in decision else "database"
    state["route_decision"] = route_decision
    state["messages"].append(AIMessage(content=f"Routing to: {route_decision.upper()}"))
    
    return state