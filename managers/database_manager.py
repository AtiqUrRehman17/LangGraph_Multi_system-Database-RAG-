import psycopg2
from psycopg2.extras import RealDictCursor
from config.settings import settings

class DatabaseManager:
    """Manages PostgreSQL database operations"""
    
    def __init__(self):
        self.connection_params = {
            'host': settings.DB_HOST,
            'port': settings.DB_PORT,
            'database': settings.DB_NAME,
            'user': settings.DB_USER,
            'password': settings.DB_PASSWORD
        }
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)
    
    def get_schema_info(self):
        """Get database schema information"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            schema_info = {}
            for (table_name,) in tables:
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