-- Migration 002: Add indexes for common queries

create index if not exists idx_users_email on public.users(email);
create index if not exists idx_users_username on public.users(username);
create index if not exists idx_users_role on public.users(role);
