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

        return redirect(url_for('quiz'))
    except Exception as e:
        logger.error(f"Error starting learning session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

# Rest of the routes (quiz, submit_answer, results, study_plan, etc.) remain unchanged

if __name__ == '__main__':
    app.run(debug=True)