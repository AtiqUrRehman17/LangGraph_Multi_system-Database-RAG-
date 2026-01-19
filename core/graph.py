from typing import Literal
from langgraph.graph import StateGraph, END
from .state import GraphState
from nodes.router_node import router_node
from nodes.rag_node import rag_node
from nodes.database_node import database_node

def route_decision(state: GraphState) -> Literal["rag", "database"]:
    """Conditional edge function to route based on decision"""
    return state["route_decision"]

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