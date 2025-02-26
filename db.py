import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("Missing Supabase credentials")
        
    return create_client(supabase_url, supabase_key)
