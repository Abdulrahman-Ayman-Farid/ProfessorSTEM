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
        self.ml_model, self.scaler = self._train_ml_model()

    def _generate_synthetic_data(self, n_samples=100):
        np.random.seed(42)
        
        # Generate more realistic data with proper distributions
        quiz_scores = np.random.normal(10, 3, n_samples).clip(0, 15)  # Mean=10, SD=3
        time_per_question = np.random.normal(90, 30, n_samples).clip(30, 180)  # Mean=90s, SD=30s
        
        # Define proficiency thresholds based on percentiles
        score_threshold_high = np.percentile(quiz_scores, 70)
        score_threshold_low = np.percentile(quiz_scores, 30)
        time_threshold_high = np.percentile(time_per_question, 30)  # Faster times are better
        time_threshold_low = np.percentile(time_per_question, 70)
        
        # Determine levels based on both score and time
        levels = []
        for score, time in zip(quiz_scores, time_per_question):
            if score >= score_threshold_high and time <= time_threshold_high:
                levels.append('high')
            elif score <= score_threshold_low or time >= time_threshold_low:
                levels.append('low')
            else:
                levels.append('intermediate')
        
        return pd.DataFrame({
            'quiz_score': quiz_scores,
            'time_per_question': time_per_question,
            'level': levels
        })

    def _train_ml_model(self):
        try:
            # Generate synthetic training data
            df = self._generate_synthetic_data()
            
            # Split features and target
            X = df[['quiz_score', 'time_per_question']]
            y = df['level']
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model with optimized hyperparameters
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy:.2f}")
            
            return model, scaler
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return None, None

    def generate_quiz(self, topic, grade):
        if not isinstance(topic, str) or not isinstance(grade, int):
            raise ValueError("Invalid topic or grade format")
            
        prompt = f"""
        Generate a 15-question MCQ Exam on {topic} for grade {grade}.
        Format each question exactly like:
        [Question X] What is the question text? | Option 1 | Option 2 | Option 3 | Option 4 | Correct Answer

        Example:
        [Question 1] What is 2+2? | 3 | 4 | 5 | 6 | 4

        Make questions progressively harder and include:
        1. Basic recall questions (33%)
        2. Understanding/application questions (33%)
        3. Analysis/problem-solving questions (34%)
        """
        return self.math_agent.generate_content(prompt)

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

    def assess_proficiency(self, correct_count, total_questions, total_time):
        try:
            if not all(isinstance(x, (int, float)) for x in [correct_count, total_questions, total_time]):
                raise ValueError("Invalid input types for proficiency assessment")
                
            if total_questions == 0:
                return "unknown"
                
            avg_time = total_time / total_questions
            
            # Scale the features
            features = np.array([[correct_count, avg_time]])
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and probabilities
            prediction = self.ml_model.predict(features_scaled)
            probabilities = self.ml_model.predict_proba(features_scaled)
            
            # Log confidence level
            confidence = np.max(probabilities)
            logger.info(f"Prediction confidence: {confidence:.2f}")
            
            return prediction[0]
        except Exception as e:
            logger.error(f"Error assessing proficiency: {str(e)}")
            return "unknown"

central_agent = CentralAgent()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start_learning', methods=['POST'])
def start_learning():
    try:
        grade = int(request.form.get('grade', 0))
        subject = request.form.get('subject', '').strip()
        topic = request.form.get('topic', '').strip()
        
        if not all([grade, subject, topic]):
            flash('Please fill in all fields', 'error')
            return redirect(url_for('home'))
            
        if grade < 1 or grade > 12:
            flash('Invalid grade level', 'error')
            return redirect(url_for('home'))
            
        session['quiz'] = {
            'grade': grade,
            'subject': subject,
            'topic': topic,
            'questions': [],
            'answers': [],
            'start_time': None,
            'current_q': 0,
            'generated': False,
        }
        return redirect(url_for('quiz'))
    except Exception as e:
        logger.error(f"Error starting learning session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/quiz')
def quiz():
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        
        if not quiz_data.get('generated'):
            quiz_raw = central_agent.generate_quiz(quiz_data['topic'], quiz_data['grade'])
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
            for grade {quiz_data['grade']} students. Format the response using this HTML structure:

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
            for grade {quiz_data['grade']} students. Format the response using this HTML structure:

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

@app.route('/final_assessment')
def final_assessment():
    if 'quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        if 'final_quiz' not in session:
            quiz_data = session['quiz']
            final_quiz = central_agent.generate_quiz(quiz_data['topic'], quiz_data['grade'])
            questions = central_agent.parse_quiz(final_quiz)
            
            if not questions:
                flash('Failed to generate final assessment questions', 'error')
                return redirect(url_for('study_plan'))
                
            session['final_quiz'] = {
                'questions': questions,
                'start_time': time.time(),
                'current_q': 0,
                'answers': []
            }
        
        final_data = session['final_quiz']
        if final_data['current_q'] >= len(final_data['questions']):
            return redirect(url_for('final_results'))
        
        current_question = final_data['questions'][final_data['current_q']]
        return render_template(
            'final_assessment.html',
            question=current_question,
            question_number=final_data['current_q'] + 1
        )
    except Exception as e:
        logger.error(f"Error displaying final assessment: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('study_plan'))

@app.route('/submit_final_answer', methods=['POST'])
def submit_final_answer():
    if 'final_quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        answer = request.form.get('answer', '').strip()
        final_data = session['final_quiz']
        
        if not answer:
            flash('Please select an answer', 'error')
            return redirect(url_for('final_assessment'))
            
        final_data['answers'].append(answer)
        final_data['current_q'] += 1
        session['final_quiz'] = final_data
        
        return redirect(url_for('final_assessment'))
    except Exception as e:
        logger.error(f"Error submitting final answer: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('final_assessment'))

@app.route('/final_results')
def final_results():
    if 'quiz' not in session or 'final_quiz' not in session:
        return redirect(url_for('home'))
    
    try:
        quiz_data = session['quiz']
        final_data = session['final_quiz']
        
        correct = sum(1 for ans, q in zip(final_data['answers'], final_data['questions']) 
                     if ans and q and ans.strip() == q["answer"].strip())
        
        total_time = time.time() - final_data['start_time']
        proficiency = central_agent.assess_proficiency(
            correct, 
            len(final_data['questions']), 
            total_time
        )
        
        initial_correct = sum(1 for ans, q in zip(quiz_data.get('answers', []), 
                                                 quiz_data.get('questions', [])) 
                            if ans and q and ans.strip() == q["answer"].strip())
        
        return render_template(
            'final_results.html',
            initial_score=f"{initial_correct}/{len(quiz_data.get('questions', []))}",
            initial_proficiency=session.get('proficiency', 'Intermediate'),
            final_score=f"{correct}/{len(final_data['questions'])}",
            final_proficiency=proficiency
        )
    except Exception as e:
        logger.error(f"Error displaying final results: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
