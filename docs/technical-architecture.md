# ProfessorSTEM — Technical Architecture

Technical documentation for the ProfessorSTEM codebase: a Flask web application that generates AI-powered quizzes and study plans for K-12 students, backed by Supabase/PostgreSQL and Google Gemini.

## System Overview

ProfessorSTEM is a server-rendered Flask web application:

- **Web layer** — `app.py` creates the Flask application and registers all routes; `wsgi.py` exposes the WSGI entry point (`from app import app`) used by Vercel. HTML is rendered from Jinja2 templates in `templates/` with static assets under `static/`.
- **Data layer** — Supabase (PostgreSQL) is accessed through the Supabase Python client (`db.py`). Schema creation and migrations are handled by `create_db.py`, `apply_migrations.py`, and the SQL files in `migrations/`.
- **AI layer** — Google Gemini (`google-generativeai`) generates quizzes and study content (`agents.py`) and assesses student proficiency (`proficiency.py`), with deterministic fallbacks when the AI is unavailable.
- **Cross-cutting** — session-based authentication with bcrypt (`auth.py`), environment-driven configuration (`config.py`, `config_agent.py`), structured JSON logging with request IDs (`logging_config.py`), response compression and security headers (`app.py`), and optional Sentry error tracking.

## Component Breakdown

### `app.py` — Application factory and HTTP surface

- Creates the Flask app and installs structured logging via `setup_logging(app)`.
- Sets the Flask `secret_key` from `config_agent.SECRET_KEY` (which reads `SECRET_KEY` from the environment).
- Configures `flask_compress` (`COMPRESS_MIMETYPES` for HTML/CSS/XML/JSON/JS, `COMPRESS_LEVEL = 6`, `COMPRESS_MIN_SIZE = 500`) for faster page loads.
- Initializes Sentry **only if** `SENTRY_DSN` is set, with the Flask integration, `traces_sample_rate=1.0`, and `environment` from `FLASK_ENV` (default `development`).
- `add_header` (`@app.after_request`) sets security headers (see *Security & Observability*) and long-lived cache headers (`Cache-Control: public, max-age=31536000`) for `/static/` paths.
- Routes: `/` (index), `/signup`, `/signin`, `/logout`, `/admin/dashboard` and `/user/dashboard` (both guarded by session role checks that redirect to home), `/todolist`, a 404 error handler rendering `404.html`, and the learning-journey routes registered with `app.add_url_rule(...)` from `routes.py`.

### `routes.py` — Learning journey flow

Implements the end-to-end student flow, with a module-level `CentralAgent` instance:

1. `user_training` — renders `user_training.html` (intake form).
2. `start_learning` (POST) — validates grade (1–12), age (5–18), school level (`Elementary School`, `Middle School`, `High School`), subject, and topic; initializes the `first_quiz` session state (questions, answers, timers, `generated` flag) and redirects to `first_quiz`.
3. `first_quiz` — lazily generates the quiz via `central_agent.generate_quiz()` / `parse_quiz()` on first visit, stores it in the session, and paginates questions 5 per page.
4. `submit_first_quiz` — collects answers and timing.
5. `first_results` — shows first-attempt results.
6. `study_plan` — generates the personalized study plan.
7. `final_quiz` / `submit_final_quiz` / `final_results` — the post-study assessment and its results.
8. `comparison` — compares first vs. final performance.

Errors are logged through the `config_agent` logger and surfaced as flash messages, redirecting back to `user_training`.

### `agents.py` — AI content generation

- Configures three Gemini clients (`gemini-1.5-flash`) using `CENTRAL_API_KEY`, `MATH_API_KEY`, and `SCIENCE_API_KEY`.
- `SubjectAgent` — wraps a `GenerativeModel`; `generate_content(prompt)` calls the model with a 0.5 s pause (rate limiting) and returns the response text, logging and returning error strings on `genai.exceptions.ApiError` or unexpected exceptions.
- `CentralAgent` — composes a `math_agent` and `science_agent`:
  - `generate_quiz(topic, grade)` validates inputs, pulls `age`/`school_level` from the Flask session (with fallbacks: `grade + 5` and `Middle School`), and builds a detailed prompt requesting a 25-question MCQ exam with a strict line format (`[Question X] text | opt1 | opt2 | opt3 | opt4 | correct`) and a 33/33/34 recall/application/analysis difficulty mix. Grades ≤ 6 use the math agent; higher grades use the science agent. If parsing yields fewer than 10 questions, it retries with the other agent.
  - `parse_quiz(quiz_text)` parses the model output into structured question data (returns `[]` for invalid input).

### `proficiency.py` — Proficiency assessment

- Configures a dedicated Gemini model, `gemini-2.0-pro-exp-02-05`, using `CENTRAL_API_KEY`.
- `assess_proficiency(correct_count, total_questions, total_time)` returns `'high'`, `'intermediate'`, `'low'` (or `'unknown'` when `total_questions == 0` or inputs are invalid).
- It computes the success percentage and average time per question, then prompts Gemini with the performance data and level guidelines (HIGH ≥ 80% with quick responses; INTERMEDIATE 50–79% or moderate pace; LOW < 50% or very slow). It retries up to 3 times (0.5 s between attempts) and accepts only the exact words `high` / `intermediate` / `low`.
- **Deterministic fallback** — if Gemini fails or returns invalid output, the level is computed directly: `high` if percentage ≥ 80 and avg time ≤ 90 s; `low` if percentage < 50 or avg time ≥ 150 s; otherwise `intermediate`.

### `auth.py` — Authentication

- `signup(request)` — hashes the password with `bcrypt.hashpw` (per-user salt), checks for an existing email in the `users` table, assigns the role `admin` to the **first registered user** and `user` to everyone else, inserts the user via Supabase, and redirects to sign-in.
- `signin(request)` — accepts an email or username (detected via a simple email regex), queries the matching user, verifies the password with `bcrypt.checkpw`, and on success stores `user`, `role`, and `user_id` in the Flask session before redirecting to the role-appropriate dashboard.
- `logout()` — clears all session data.

### `db.py` — Supabase client

`get_db_connection()` builds and returns a Supabase client from `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`, raising `ValueError("Missing Supabase credentials")` if either is unset. All table access (users) goes through this client.

### `config.py` / `config_agent.py` — Configuration

- `config.py` — loads `.env` via `python-dotenv` and exposes `SECRET_KEY` plus legacy `DB_HOST` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` / `DB_PORT` variables.
- `config_agent.py` — loads `.env`, defines a module-level `logger` (INFO level), a hardcoded `SECRET_KEY` fallback used by `app.py`, and the three Gemini API keys (`CENTRAL_API_KEY`, `MATH_API_KEY`, `SCIENCE_API_KEY`), raising `ValueError("API keys not found in environment variables.")` if any is missing.

### `logging_config.py` — Structured JSON logging

- `StructuredFormatter` emits one JSON object per log record with `timestamp` (UTC ISO-8601), `level`, `message`, `module`, `function`, `line`, and — when available from the Flask `g` object — `request_id`, `route`, `method`, `status_code`, `latency_ms`, plus formatted exceptions.
- `setup_logging(app)` replaces `app.logger` handlers with a `StreamHandler` using this formatter (INFO level), and installs middleware: `assign_request_id` (`@app.before_request`) generates a UUID `request_id` and records route/method/start time; `log_request` (`@app.after_request`) computes latency in milliseconds and logs every completed request.

### `create_db.py` / `apply_migrations.py` / `migrations/` — Schema and migrations

- `create_db.py` — connects with `psycopg2` using `POSTGRES_URL` (parsed with `urlparse`, `sslmode='require'`) and creates the `public.users` table with RLS and its two policies (idempotent `create table if not exists` / `drop policy if exists`).
- `apply_migrations.py` — creates the `public.migrations` tracking table, then applies every `migrations/*.sql` file in sorted order that has not been applied yet, recording each in the tracking table and committing per file (rollback + re-raise on failure). Re-running skips already-applied files, making it idempotent.
- `migrations/` — `001_initial_schema.sql` (users table + RLS) and `002_add_indexes.sql`, with a `README.md`.

### `optimize_assets.py` — Asset minification CLI

A command-line utility that minifies `static/css/style.css` (with `cssmin`) and `static/js/bundle.js` (with `jsmin`) **in place**, printing before/after byte sizes and the percentage reduction. Run with `python optimize_assets.py`.

## Data Model

### `public.users` (from `migrations/001_initial_schema.sql`)

```sql
create table if not exists public.users (
    id uuid default gen_random_uuid() primary key,
    username text unique not null,
    full_name text not null,
    email text unique not null,
    password text not null,
    role text not null check (role in ('admin', 'user')),
    created_at timestamp with time zone default timezone('utc'::text, now())
);
```

Row Level Security is enabled on the table with two policies:

- **"Users can view own data"** — `for select using (auth.uid() = id)`
- **"Users can update own data"** — `for update using (auth.uid() = id)`

### `public.migrations` (from `apply_migrations.py`)

Tracking table for applied migration files:

```sql
create table if not exists public.migrations (
    id serial primary key,
    filename text unique not null,
    applied_at timestamp with time zone default timezone('utc'::text, now())
);
```

## AI Integration Design

- **Models** — `gemini-1.5-flash` is used for quiz and content generation (three separately configured clients in `agents.py`); `gemini-2.0-pro-exp-02-05` is used for proficiency assessment (`proficiency.py`).
- **API keys** — three Gemini keys are required and validated at import time in `config_agent.py`:
  - `CENTRAL_API_KEY` — main key, used by the proficiency model and the central client.
  - `MATH_API_KEY` — key for the math content agent.
  - `SCIENCE_API_KEY` — key for the science content agent.
- **Prompt structure (quiz generation)** — the prompt in `CentralAgent.generate_quiz` requests a 25-question MCQ exam tailored to topic, grade, age, and school level, emphasizing age-appropriate language, curriculum standards, attention span, and relatable real-world examples. It enforces an exact output format per line:

  ```text
  [Question X] What is the question text? | Option 1 | Option 2 | Option 3 | Option 4 | Correct Answer
  ```

  and a difficulty mix of ~33% recall, ~33% understanding/application, ~34% analysis/problem-solving, with questions ordered progressively harder.
- **Prompt structure (proficiency)** — the prompt presents correct count, success rate, average time per question, and total duration, plus level guidelines, and demands a single-word answer (`high`, `intermediate`, or `low`).
- **Resilience** — quiz generation retries with the alternate subject agent when fewer than 10 questions parse; proficiency assessment retries up to 3 times and then falls back to the deterministic thresholds described above. All model calls include a 0.5 s delay for rate limiting.

## Security & Observability

- **Security headers** (`app.py`, `add_header`):
  - `Content-Security-Policy`: `default-src 'self'`; scripts/styles allow `'self'`, `'unsafe-inline'`, and `https://cdn.tailwindcss.com`; images allow `'self'`, `data:`, `https:`; fonts allow `'self'`, `data:`; `connect-src 'self' https://*.supabase.co`.
  - `Strict-Transport-Security`: `max-age=31536000; includeSubDomains`.
  - `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`.
- **Password hashing** — bcrypt with per-user salts in `auth.py` (`hashpw` on signup, `checkpw` on sign-in); plaintext passwords are never stored.
- **Role-based access** — the first registered user becomes `admin`; dashboard routes check `session['role']` and redirect unauthorized users to home.
- **Structured logging** — JSON logs with a per-request UUID `request_id`, route, method, status code, and latency (see `logging_config.py`).
- **Sentry** — enabled only when `SENTRY_DSN` is set, with the Flask integration and full tracing (`traces_sample_rate=1.0`), environment from `FLASK_ENV`.
- **Health check** — the `/api/health` endpoint is exposed for uptime/liveness monitoring.

## Environment & Configuration

All configuration is environment-driven, loaded via `python-dotenv` (`load_dotenv()` in `app.py`, `db.py`, `config.py`, `config_agent.py`, etc.). `.env.example` documents the full set:

| Variable | Purpose |
| --- | --- |
| `POSTGRES_URL` | Pooled Supabase Postgres connection URL (used by `create_db.py` / `apply_migrations.py`) |
| `POSTGRES_URL_NON_POOLING` | Non-pooled Supabase Postgres URL |
| `POSTGRES_PRISMA_URL` | Prisma-compatible pooled Postgres URL |
| `SUPABASE_URL` | Supabase project URL (used by `db.py`) |
| `NEXT_PUBLIC_SUPABASE_URL` | Public Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Public Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Service-role key (used by `db.py`) |
| `SUPABASE_JWT_SECRET` | Supabase JWT secret |
| `POSTGRES_HOST` | Direct database host |
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_DATABASE` | Database name |
| `SECRET_KEY` | Flask session signing key |
| `CENTRAL_API_KEY` | Main Gemini API key (proficiency + central client) |
| `MATH_API_KEY` | Gemini API key for math content generation |
| `SCIENCE_API_KEY` | Gemini API key for science content generation |
| `SENTRY_DSN` | Sentry DSN; when set, Sentry error tracking is enabled |
| `FLASK_ENV` | Environment name reported to Sentry (default `development`) |

> Note: `.env.example` instructs replacing all `[YOUR-*]` placeholders and never committing real values to version control.

## Deployment

- **Vercel** — `vercel.json` declares `version: 2`, builds `wsgi.py` with the `@vercel/python` builder, and routes all paths (`/(.*)`) to `wsgi.py`. `wsgi.py` simply imports the Flask `app` from `app.py`, making the entire application available as a serverless Python function.
- **CI/CD** — `.github/workflows/ci-cd.yml` runs on pushes to `main` and on all pull requests, with read-only permissions and per-ref concurrency cancellation:
  - `quality` — Ubuntu runner with an ephemeral **PostgreSQL 15** service container (`professorstem_test` database, health-checked via `pg_isready`); installs dependencies plus `pytest`/`flake8`, runs a non-blocking lint (`E9,F63,F7,F82`), runs `pytest tests/ -v` against the ephemeral database, and performs a `python -m compileall .` build check.
  - `deploy-preview` — runs for pull requests after `quality` passes, using `VERCEL_TOKEN` / `VERCEL_ORG_ID` / `VERCEL_PROJECT_ID` secrets (currently a placeholder deployment step).
  - `deploy-production` — runs for pushes to `main` after `quality` passes, using `DEPLOY_TOKEN` / `DEPLOY_HOST` secrets.
- **Backup & recovery** — see [docs/database-backup.md](database-backup.md) for backup and recovery procedures.

## Local Development Setup

**Prerequisites:** Python 3.7+, `pip`, and a PostgreSQL database (Supabase project or local Postgres).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# then edit .env and replace all [YOUR-*] placeholders

# 3. Create the database schema (users table + RLS)
python create_db.py

# 4. Apply migrations (idempotent)
python apply_migrations.py

# 5. Run the development server
flask run --host=0.0.0.0 --port=5000
```

The app is then available at `http://localhost:5000`. The first account registered through `/signup` becomes the `admin`.
