import os
import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def test_db_connection():
    try:
        # Get Supabase database URL
        db_url = os.getenv('POSTGRES_URL')
        
        if not db_url:
            raise ValueError("Missing POSTGRES_URL in environment variables")

        # Parse the connection details from the URL
        parsed = urlparse(db_url)
        dbname = parsed.path[1:]  # Remove leading slash
        user = parsed.username
        password = parsed.password
        host = parsed.hostname
        port = parsed.port

        # Connect to the database
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            sslmode='require'
        )
        
        print("✅ Connected to PostgreSQL database successfully!")
        
        # Test querying the users table
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM public.users")
        count = cur.fetchone()[0]
        print(f"✅ Found {count} users in the database")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    test_db_connection()
