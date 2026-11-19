# Database Migrations

This directory contains versioned SQL migration files for the ProfessorSTEM database.

## How to Apply Migrations

Migrations are applied in numerical order. To apply all migrations:

```bash
python apply_migrations.py
```

## How to Create a New Migration

1. Create a new file with the next sequential number: `migrations/003_description.sql`
2. Write your SQL changes in the file
3. Test the migration against a local database
4. Commit the file to version control

## Rollback Strategy

For destructive changes, always create a corresponding rollback file:
- `migrations/003_add_column.sql`
- `migrations/003_add_column_rollback.sql`

The rollback file should contain the SQL to reverse the changes.
