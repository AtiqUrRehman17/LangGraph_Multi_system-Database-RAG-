from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """State for the LangGraph workflow"""
    query: str
    route_decision: str
    rag_response: str
    sql_query: str
    sql_result: str
    final_response: str
    error: str
    messages: Annotated[list[BaseMessage], operator.add]