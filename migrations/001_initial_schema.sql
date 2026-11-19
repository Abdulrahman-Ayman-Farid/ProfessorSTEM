-- Migration 001: Initial schema
-- Creates the users table with RLS policies

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
