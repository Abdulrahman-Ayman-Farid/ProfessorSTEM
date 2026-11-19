import pytest
import os
from db import get_db_connection

pytestmark = pytest.mark.skipif(
    not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_SERVICE_ROLE_KEY'),
    reason='Supabase credentials not configured'
)

def test_db_connection():
    """Test that database connection can be established."""
    try:
        supabase = get_db_connection()
        assert supabase is not None
    except Exception as e:
        pytest.fail(f"Database connection failed: {str(e)}")

def test_users_table_exists():
    """Test that the users table exists in the database."""
    try:
        supabase = get_db_connection()
        response = supabase.table('users').select("*").limit(1).execute()
        assert response is not None
    except Exception as e:
        pytest.fail(f"Users table check failed: {str(e)}")
