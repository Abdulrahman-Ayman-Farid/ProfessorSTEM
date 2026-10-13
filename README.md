# ProfessorSTEM: AI Learning Companion

ProfessorSTEM is a Flask-based educational platform leveraging AI agents to generate quizzes, assess student proficiency, and provide personalized study plans in STEM subjects.

## Features

- **User Authentication** — Role-based access control with Admin and User roles for secure session management.
- **AI-Driven Quiz Generation** — Automatically generates subject-specific quizzes using external AI APIs.
- **Proficiency Assessment** — Evaluates student knowledge levels and tracks progress over time.
- **Progress Comparison** — Compares initial and final assessment results to visualize learning gains.
- **Admin Dashboard** — Centralized interface for managing users, viewing analytics, and overseeing platform activity.

## Tech Stack

- **Python** — Core application logic
- **Flask** — Web framework for routing and server-side rendering
- **PostgreSQL** — Relational database for persistent storage
- **Tailwind CSS** — Utility-first CSS framework for styling
- **Alpine.js** — Lightweight JavaScript framework for frontend interactivity
- **External AI APIs** — Gemini AI integration for content generation and assessment

## Project Structure

```
ProfessorSTEM/
├── app.py              # Flask application entry point
├── routes.py           # Application route definitions
├── auth.py             # Authentication and authorization logic
├── agents.py           # AI agent integrations for quiz and content generation
├── proficiency.py      # Proficiency assessment and scoring logic
├── db.py               # Database connection and query helpers
├── create_db.py        # Database initialization script
├── config.py           # Application configuration
├── config_agent.py     # AI agent configuration
├── templates/          # Jinja2 HTML templates
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # Project documentation
```

## Prerequisites

- **Python 3.8+**
- **PostgreSQL**
- **pip**

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/ProfessorSTEM.git
   cd ProfessorSTEM
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Linux/Mac:**

     ```bash
     source venv/bin/activate
     ```

   - **Windows:**

     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` and provide values for the following variables:

| Variable           | Description                                      |
| ------------------ | ------------------------------------------------ |
| `SECRET_KEY`       | Flask session secret key                         |
| `DB_HOST`          | PostgreSQL database host                         |
| `DB_NAME`          | PostgreSQL database name                         |
| `DB_USER`          | PostgreSQL database user                         |
| `DB_PASSWORD`      | PostgreSQL database password                     |
| `DB_PORT`          | PostgreSQL database port                         |
| `CENTRAL_API_KEY`  | Gemini API key for proficiency assessment        |
| `MATH_API_KEY`     | Gemini API key for math content generation       |
| `SCIENCE_API_KEY`  | Gemini API key for science content generation    |

## Database Setup

Initialize the database tables by running:

```bash
python create_db.py
```

This script creates all required tables in your PostgreSQL database.

## Running the Application

Start the Flask development server:

```bash
flask run
```

Alternatively, run the application directly:

```bash
python app.py
```

The application will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).
