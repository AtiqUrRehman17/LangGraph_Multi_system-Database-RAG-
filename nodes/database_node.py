from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import GraphState
from managers.database_manager import DatabaseManager
from config.settings import settings

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    openai_api_key=settings.OPENAI_API_KEY
)

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
        
        # Format schema
        schema_text = ""
        for table, columns in schema_info.items():
            schema_text += f"\nTable: {table}\n"
            for col_name, col_type in columns:
                schema_text += f"  - {col_name} ({col_type})\n"
        
        # Generate SQL
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Generate a PostgreSQL query for the user's question.

Database Schema:
{schema}

Return ONLY the SQL query, no explanations."""),
            ("human", "{query}")
        ])
        
        chain = sql_prompt | llm | StrOutputParser()
        sql_query = chain.invoke({"schema": schema_text, "query": query}).strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        state["sql_query"] = sql_query
        
        # Execute query
        results = db_manager.execute_query(sql_query)
        
        if isinstance(results, dict) and "error" in results:
            error_msg = f"SQL error: {results['error']}"
            state["final_response"] = error_msg
        else:
            state["sql_result"] = str(results)
            
            # Generate explanation
            result_prompt = ChatPromptTemplate.from_messages([
                ("system", """Explain the database query results in natural language.

SQL Query: {sql_query}
Results: {results}"""),
                ("human", "Explain these results clearly and concisely.")
            ])
            
            chain = result_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "sql_query": sql_query,
                "results": str(results)
            })
            
            state["final_response"] = response
        
        state["messages"].append(AIMessage(content=state["final_response"]))
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        state["error"] = error_msg
        state["final_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
    
    return state