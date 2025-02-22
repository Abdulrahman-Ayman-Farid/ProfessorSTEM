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

    def generate_quiz(self, topic, level, subject):
        if not isinstance(topic, str) or not isinstance(level, str) or not isinstance(subject, str):
            raise ValueError("Invalid topic, level, or subject format")
            
        prompt = f"""
        Generate a 15-question MCQ Exam on {topic} for {level} level students in {subject}.
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
        school_level = request.form.get('school_level'))
        grade = int(request.form.get('grade'))
        subject = request.form.get('subject'))
        topic = request.form.get('topic'))

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

        return redirect(url_for('quiz'))  # Redirect to the quiz route
    except Exception as e:
        logger.error(f"Error starting learning session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@@app.route('/quiz')
def quiz():
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        
        if not quiz_data.get('generated'):
            quiz_raw = central_agent.generate_quiz(quiz_data['topic'], quiz_data['level'], quiz_data['subject'])
            questions = central_agent.parse_quiz(quiz_raw)
            
            if not questions:
                flash('Failed to generate quiz questions', 'error')
                return redirect(url_for('home'))
                
            quiz_data.update({
                "questions": questions,
                "start_time": time.time(),
                "current_q": 0,
                "generated": True
            })
            session['quiz'] = quiz_data

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
        answers_dict = {}
        
        for key, value in request.form.items():
            if key.startswith('answers[') and key.endswith(']'):
                try:
                    idx = int(key[8:-1])
                    answers_dict[idx] = value.strip()
                except ValueError:
                    continue
        
        current_answers = quiz_data.get('answers', [])
        for idx, answer in answers_dict.items():
            while len(current_answers) <= idx:
                current_answers.append(None)
            current_answers[idx] = answer
        
        quiz_data['answers'] = current_answers
        session['quiz'] = quiz_data

        current_page = quiz_data['current_q'] // 5
        total_questions = len(quiz_data['questions'])
        total_pages = (total_questions + 4) // 5

        if current_page >= total_pages - 1:
            return redirect(url_for('results'))
        else:
            quiz_data['current_q'] = (current_page + 1) * 5
            session['quiz'] = quiz_data
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
        correct = sum(1 for ans, q in zip(quiz_data['answers'], quiz_data['questions']) 
                     if ans and q and ans.strip() == q["answer"].strip())
        
        total_time = time.time() - quiz_data['start_time']
        proficiency = central_agent.assess_proficiency(
            correct, 
            len(quiz_data['questions']), 
            total_time
        )
        
        session['proficiency'] = proficiency
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
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        subject = quiz_data['subject']
        agent = central_agent.math_agent if subject == "Math" else central_agent.science_agent
        
        study_plan = agent.generate_content(f"""
            Create a study plan for {quiz_data['topic']} ({session.get('proficiency', 'intermediate')} level) 
            for Level {quiz_data['Level']} students. Format the response using this HTML structure:

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
                
                <div class="practice-exercises">
                    <strong>Practice Activities:</strong>
                    <ul class="study-list">
                        <li>Exercise 1</li>
                        <li>Exercise 2</li>
                    </ul>
                </div>
                
                <div class="milestone">
                    <strong>Progress Check:</strong>
                    What you should know by now...
                </div>
            </div>

            Create 3-4 topic sections, with clear progression from basic to advanced concepts.
        """)
        
        guide = agent.generate_content(f"""
            Create a detailed guide for {quiz_data['topic']} in {subject} 
            for Level {quiz_data['Level']} students. Format the response using this HTML structure:

            <div class="topic-section">
                <div class="topic-title">Concept Overview</div>
                
                <div class="key-concept">
                    <strong>Key Points:</strong>
                    <ul class="study-list">
                        <li>Main point 1</li>
                        <li>Main point 2</li>
                    </ul>
                </div>

                <div class="subtopic-title">Detailed Explanation</div>
                <p>Explanation text goes here...</p>
                
                <div class="example-box">
                    <strong>Example:</strong>
                    <p>Step 1: ...</p>
                    <p>Step 2: ...</p>
                </div>
            </div>

            Include multiple sections covering core concepts, common mistakes to avoid, 
            and solved examples with clear explanations.
        """)
        
        return render_template('study_plan.html', study_plan=study_plan, guide=guide)
    except Exception as e:
        logger.error(f"Error generating study plan: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)