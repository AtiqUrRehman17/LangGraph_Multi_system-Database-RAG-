import psycopg2
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

try:
    # Connect using environment variables
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    
    print("✅ Connected to PostgreSQL successfully!")
    
    # Quick test query
    cur = conn.cursor()
    cur.execute("SELECT 'Database is working!' as status, NOW() as time")
    result = cur.fetchone()
    print(f"📊 {result[0]}")
    print(f"🕐 Time: {result[1]}")
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")