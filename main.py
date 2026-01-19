from langchain_core.messages import HumanMessage
from core.graph import build_graph
from managers import PineconeManager, DatabaseManager

class RAGDatabaseApp:
    """Main application class"""
    
    def __init__(self):
        self.graph = build_graph()
        self.pinecone_manager = PineconeManager()
        self.db_manager = DatabaseManager()
    
    def upload_pdf(self, pdf_path: str) -> str:
        """Upload a PDF to the RAG system"""
        return self.pinecone_manager.upload_pdf(pdf_path)
    
    def query(self, user_query: str) -> dict:
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
        
        return self.graph.invoke(initial_state)


def main():
    """Main entry point"""
    app = RAGDatabaseApp()
    
    print("=" * 60)
    print("LangGraph RAG & Text-to-SQL Router System")
    print("=" * 60)
    print("\nCommands:")
    print("  'exit'   - Quit")
    print("  'upload' - Upload PDF")
    print("=" * 60)
    
    while True:
        user_input = input("\nQuery: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'upload':
            pdf_path = input("PDF path: ").strip()
            result = app.upload_pdf(pdf_path)
            print(f"\n{result}")
            continue
        
        if not user_input:
            continue
        
        # Process query
        result = app.query(user_input)
        print(f"\n{result['final_response']}")


if __name__ == "__main__":
    main()