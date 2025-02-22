import os
import time
import logging
import pandas as pd
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, session, redirect, url_for, flash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = '16a4a22af5c3ffa329130ffb50b64d75bf2733e370844174dea6ab456a0da06a'
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# Load environment variables
load_dotenv()

# Load the saved model pipeline
try:
    with open("model_pipeline.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
    logger.info("Model pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model pipeline: {str(e)}")
    raise

# Configure Gemini clients
try:
    central_api_key = os.getenv("CENTRAL_API_KEY")
    math_api_key = os.getenv("MATH_API_KEY")
    science_api_key = os.getenv("SCIENCE_API_KEY")
    if not central_api_key or not math_api_key or not science_api_key:
        raise ValueError("API keys not found in environment variables.")

    genai.configure(api_key=central_api_key)
    central_model = genai.GenerativeModel('gemini-1.5-pro-latest')

    genai.configure(api_key=math_api_key)
    math_model = genai.GenerativeModel('gemini-1.5-pro-latest')

    genai.configure(api_key=science_api_key)
    science_model = genai.GenerativeModel('gemini-1.5-pro-latest')

except Exception as e:
    logger.error(f"Error initializing Generative AI models: {str(e)}")
    raise

class SubjectAgent:
    def __init__(self, model):
        self.model = model
        
    def generate_content(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            time.sleep(0.5)
            return response.text
        except genai.exceptions.ApiError as e:
            logger.error(f"API Error: {str(e)}")
            return f"API Error: {e}"
        except Exception as e:
            logger.error(f"Unexpected Error: {str(e)}")
            return f"Unexpected Error: {str(e)}"

class CentralAgent:
    def __init__(self):
        self.math_agent = SubjectAgent(math_model)
        self.science_agent = SubjectAgent(science_model)
    
    def assess_proficiency(self, correct, total, time_taken):
        score_percentage = (correct / total) * 100
        avg_time_per_q = time_taken / total
        
        if score_percentage >= 85 and avg_time_per_q <= 45:
            return "advanced"
        elif score_percentage >= 70 or (score_percentage >= 60 and avg_time_per_q <= 60):
            return "intermediate"
        else:
            return "beginner"

    def generate_quiz(self, topic, level, subject):
        if not isinstance(topic, str) or not isinstance(level, str) or not isinstance(subject, str):
            raise ValueError("Invalid topic, level, or subject format")
            
        prompt = f"""
        Generate a 25-question MCQ Exam on {topic} for {level} level students in {subject}.
        Format each question exactly like:
        [Question X] What is the question text? | Option 1 | Option 2 | Option 3 | Option 4 | Correct Answer

        Example:
        [Question 1] What is 2+2? | 3 | 4 | 5 | 6 | 4

        Make questions progressively harder and include:
        1. Basic recall questions (33%)
        2. Understanding/application questions (33%)
        3. Analysis/problem-solving questions (34%)
        """
        return self.math_agent.generate_content(prompt) if subject == "Math" else self.science_agent.generate_content(prompt)

    def parse_quiz(self, quiz_text):
        if not isinstance(quiz_text, str):
            logger.error("Invalid quiz text format")
            return []
            
        try:
            questions = []
            for line in quiz_text.split("\n"):
                if line.strip() and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 6:
                        question_text = parts[0]
                        if "[Question" in question_text:
                            question_text = question_text.split("]", 1)[1].strip()
                        
                        questions.append({
                            "question": question_text,
                            "options": parts[1:-1],
                            "answer": parts[-1]
                        })
            return questions
        except Exception as e:
            logger.error(f"Error parsing quiz: {str(e)}")
            return []

central_agent = CentralAgent()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_level', methods=['POST'])
def predict_level():
    try:
        # Get user inputs
        age = int(request.form.get('age'))
        school_level = request.form.get('school_level')
        grade = int(request.form.get('grade'))
        subject = request.form.get('subject')
        topic = request.form.get('topic')
        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            "Age": [age],
            "School_Level": [school_level],
            "Grade": [grade]
        })

        # Predict the Level using the loaded model pipeline
        predicted_level = model_pipeline.predict(input_data)[0]
        logger.info(f"Predicted Level: {predicted_level}")

        # Render the home template with the predicted level, subject, and topic
        return render_template('home.html', 
                               predicted_level=predicted_level, 
                               subject=subject, 
                               topic=topic)
    except Exception as e:
        logger.error(f"Error predicting level: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/start_learning', methods=['POST'])
def start_learning():
    try:
        level = request.form.get('level')
        subject = request.form.get('subject')
        topic = request.form.get('topic')

        if not all([level, subject, topic]):
            flash('Please fill in all fields', 'error')
            return redirect(url_for('home'))

        # Generate the quiz using the subject, topic, and level
        quiz_raw = central_agent.generate_quiz(topic, level, subject)
        questions = central_agent.parse_quiz(quiz_raw)

        if not questions:
            flash('Failed to generate quiz questions', 'error')
            return redirect(url_for('home'))

        # Store quiz data in the session
        session['quiz'] = {
            'level': level,
            'subject': subject,
            'topic': topic,
            'questions': questions,
            'answers': [],
            'start_time': time.time(),
            'current_q': 0,
            'generated': True
        }
        session.modified = True

        return redirect(url_for('quiz'))
    except Exception as e:
        logger.error(f"Error starting learning session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/quiz')
@app.route('/quiz/<int:page>')
def quiz(page=None):
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        total_questions = len(quiz_data.get('questions', []))
        
        logger.info(f"Quiz page request - Page: {page}, Current Q: {quiz_data.get('current_q', 0)}")
        
        # Validate current_q is within bounds
        if quiz_data.get('current_q', 0) >= total_questions:
            quiz_data['current_q'] = 0
            
        # If page is explicitly provided, update current_q
        if page is not None:
            new_current_q = page * 5
            if new_current_q < total_questions:
                quiz_data['current_q'] = new_current_q
                session['quiz'] = quiz_data
                session.modified = True
                logger.info(f"Updated current_q to {new_current_q}")
        
        questions = quiz_data.get('questions', [])
        current_page = quiz_data['current_q'] // 5
        start_idx = current_page * 5
        end_idx = min((current_page + 1) * 5, len(questions))
        current_questions = questions[start_idx:end_idx]
        
        return render_template(
            'quiz.html',
            questions=current_questions,
            current_page=current_page,
            total_pages=(len(questions) + 4) // 5,
            start_idx=start_idx
        )
    except Exception as e:
        logger.error(f"Error displaying quiz: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        action = request.form.get('action', 'next')
        answers_dict = {}
        
        # Process submitted answers
        for key, value in request.form.items():
            if key.startswith('answers[') and key.endswith(']'):
                try:
                    idx = int(key[8:-1])
                    answers_dict[idx] = value.strip()
                except ValueError:
                    continue
        
        # Update answers in session
        total_questions = len(quiz_data['questions'])
        current_answers = quiz_data.get('answers', [None] * total_questions)
        
        # Ensure answers array is sized correctly
        if len(current_answers) < total_questions:
            current_answers.extend([None] * (total_questions - len(current_answers)))
        
        logger.info(f"Processing answers - Current answers count: {len(current_answers)}")
        logger.info(f"New answers being submitted: {answers_dict}")
        
        # Update answers
        for idx, answer in answers_dict.items():
            if idx < total_questions:
                current_answers[idx] = answer.strip() if answer else None
        
        # Update quiz data and session
        quiz_data['answers'] = current_answers
        session['quiz'] = quiz_data
        session.modified = True
        
        # Calculate navigation
        current_page = quiz_data['current_q'] // 5
        total_pages = (total_questions + 4) // 5
        
        logger.info(f"Navigation - Action: {action}, Current Page: {current_page}/{total_pages}")
        
        # Handle navigation
        if action == 'previous' and current_page > 0:
            logger.info("Moving to previous page")
            quiz_data['current_q'] = (current_page - 1) * 5
            session['quiz'] = quiz_data
            return redirect(url_for('quiz'))
            
        if action == 'complete' or (action == 'next' and current_page >= total_pages - 1):
            logger.info("Completing quiz and moving to results")
            return redirect(url_for('results'))
            
        if action == 'next':
            logger.info("Moving to next page")
            quiz_data['current_q'] = (current_page + 1) * 5
            session['quiz'] = quiz_data
            return redirect(url_for('quiz'))
        
        # Default: stay on current page
        logger.warning(f"Unexpected action: {action}")
        return redirect(url_for('quiz'))
        
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('quiz'))

@app.route('/results')
def results():
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        correct = sum(1 for ans, q in zip(quiz_data.get('answers', []), quiz_data.get('questions', [])) 
                     if ans and q and ans.strip() == q.get("answer", "").strip())
        
        # Ensure values are valid for template calculations
        correct = max(0, correct)  # Prevent negative values
        total = len(quiz_data.get('questions', []))
        if total == 0:
            flash('Quiz data is invalid', 'error')
            return redirect(url_for('home'))
        
        try:
            total_time = time.time() - float(quiz_data.get('start_time', 0))
            total_time = max(0, total_time)  # Ensure non-negative
        except (TypeError, ValueError):
            total_time = 0
            logger.error("Invalid start_time in quiz data")

        total_questions = len(quiz_data.get('questions', []))
        if total_questions > 0:
            proficiency = central_agent.assess_proficiency(
                correct,
                total_questions,
                total_time
            )
        else:
            proficiency = "beginner"
            logger.warning("No questions found in quiz data")
        
        # Update session with quiz results
        quiz_data['proficiency'] = proficiency
        quiz_data['correct'] = correct
        quiz_data['total_time'] = int(total_time)
        session['quiz'] = quiz_data
        session.modified = True
        
        return render_template(
            'results.html',
            correct=correct,
            total=len(quiz_data['questions']),
            time_taken=int(total_time),
            proficiency=proficiency
        )
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/study_plan')
def study_plan():
    try:
        logger.info("Study plan route accessed")
        
        if 'quiz' not in session:
            logger.warning("No quiz data in session")
            flash('Please start a quiz first', 'warning')
            return redirect(url_for('home'))
        
        quiz_data = session['quiz']
        logger.info(f"Quiz data keys in session: {quiz_data.keys()}")
        
        # Check if quiz is completed
        if not quiz_data.get('proficiency') or not quiz_data.get('correct'):
            logger.warning("Quiz not completed or results not available")
            flash('Please complete the quiz first', 'warning')
            return redirect(url_for('results'))
            
        subject = quiz_data['subject']
        agent = central_agent.math_agent if subject == "Math" else central_agent.science_agent
        
        logger.info(f"Generating study plan for topic: {quiz_data['topic']}, level: {quiz_data['level']}, proficiency: {quiz_data['proficiency']}")
        
        study_plan = agent.generate_content(f"""
            Create a study plan for {quiz_data['topic']} ({quiz_data['proficiency']} level) 
            for Level {quiz_data['level']} students. Format the response using this HTML structure:

            <div class="topic-section">
                <div class="topic-title">Topic Name (Duration)</div>
                <div class="learning-objectives">
                    <strong>Learning Goals:</strong>
                    <ul class="study-list">
                        <li>Goal 1</li>
                        <li>Goal 2</li>
                    </ul>
                </div>
                <div class="subtopic-title">Subtopic 1</div>
                <ul class="study-list">
                    <li>Key point 1</li>
                    <li>Key point 2</li>
                </ul>
            </div>
        """)
        
        guide = agent.generate_content(f"""
            Create a detailed guide for {quiz_data['topic']} in {subject}. 
            Focus on {quiz_data['proficiency']} level concepts. Format as HTML:

            <div class="guide-section">
                <h3>Key Concepts</h3>
                <ul>
                    <li>Concept 1: explanation...</li>
                    <li>Concept 2: explanation...</li>
                </ul>
                <h3>Examples</h3>
                <div class="example">
                    <p>Problem: ...</p>
                    <p>Solution: ...</p>
                </div>
            </div>
        """)
        
        # Pass additional context data to template
        context = {
            'study_plan': study_plan,
            'guide': guide,
            'quiz_data': {
                'topic': quiz_data['topic'],
                'level': quiz_data['level'],
                'proficiency': quiz_data['proficiency'],
                'correct': quiz_data['correct'],
                'total': len(quiz_data['questions']),
                'time_taken': quiz_data['total_time']
            }
        }
        
        logger.info("Successfully generated study plan and guide")
        return render_template('study_plan.html', **context)
    except Exception as e:
        logger.error(f"Error generating study plan: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)
