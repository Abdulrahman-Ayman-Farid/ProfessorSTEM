import os
import glob
import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

def get_connection():
    db_url = os.getenv('POSTGRES_URL')
    if not db_url:
        raise ValueError("Missing POSTGRES_URL in environment variables")
    
    parsed = urlparse(db_url)
    return psycopg2.connect(
        dbname=parsed.path[1:],
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port,
        sslmode='require'
    )

def apply_migrations():
    conn = get_connection()
    cur = conn.cursor()
    
    # Create migrations tracking table
    cur.execute("""
        create table if not exists public.migrations (
            id serial primary key,
            filename text unique not null,
            applied_at timestamp with time zone default timezone('utc'::text, now())
        );
    """)
    conn.commit()
    
    # Get already applied migrations
    cur.execute("SELECT filename FROM public.migrations")
    applied = {row[0] for row in cur.fetchall()}
    
    # Get all migration files
    migration_files = sorted(glob.glob('migrations/*.sql'))
    
    for file_path in migration_files:
        filename = os.path.basename(file_path)
        if filename in applied:
            print(f"Skipping {filename} (already applied)")
            continue
        
        print(f"Applying {filename}...")
        with open(file_path, 'r') as f:
            sql = f.read()
        
        try:
            cur.execute(sql)
            cur.execute("INSERT INTO public.migrations (filename) VALUES (%s)", (filename,))
            conn.commit()
            print(f"  ✓ {filename} applied successfully")
        except Exception as e:
            conn.rollback()
            print(f"  ✗ {filename} failed: {str(e)}")
            raise
    
    cur.close()
    conn.close()
    print("All migrations applied successfully!")

if __name__ == "__main__":
    apply_migrations()
