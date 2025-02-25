import os
import requests
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def create_tables():
    try:
        # Get Supabase database URL
        db_url = os.getenv('POSTGRES_URL')
        
        if not db_url:
            raise ValueError("Missing database URL in environment variables")

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
        
        # Create a cursor
        cur = conn.cursor()

        # SQL query to create users table with RLS
        sql_query = """
        create table if not exists public.users (
            id uuid default gen_random_uuid() primary key,
            username text unique not null,
            full_name text not null,
            email text unique not null,
            password text not null,
            role text not null check (role in ('admin', 'user')),
            created_at timestamp with time zone default timezone('utc'::text, now())
        );

        alter table public.users enable row level security;

        drop policy if exists "Users can view own data" on users;
        drop policy if exists "Users can update own data" on users;

        create policy "Users can view own data" on public.users
            for select using (auth.uid() = id);

        create policy "Users can update own data" on public.users
            for update using (auth.uid() = id);
        """

        # Execute the query
        cur.execute(sql_query)
        
        # Commit the changes
        conn.commit()
        
        print("Users table created successfully!")
        return True

    except Exception as e:
        print(f"Error creating users table: {e}")
        return False
    
    finally:
        # Close cursor and connection
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

def main():
    print("Starting database setup...")
    success = create_tables()
    if success:
        print("Database setup completed successfully!")
    else:
        print("Database setup failed!")

if __name__ == "__main__":
    main()
