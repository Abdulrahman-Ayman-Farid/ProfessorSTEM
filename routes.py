import time
from flask import render_template, request, session, redirect, url_for, flash
from config_agent import logger
from agents import CentralAgent

central_agent = CentralAgent()

def user_training():
    return render_template('user_training.html')

def start_learning():
    try:
        grade = int(request.form.get('grade', 0))
        subject = request.form.get('subject', '').strip()
        topic = request.form.get('topic', '').strip()
        
        if not all([grade, subject, topic]):
            flash('Please fill in all fields', 'error')
            return redirect(url_for('user_training'))
            
        if grade < 1 or grade > 12:
            flash('Invalid grade level', 'error')
            return redirect(url_for('user_training'))
            
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
        return redirect(url_for('user_training'))

def quiz():
    if 'quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['quiz']
        
        if not quiz_data.get('generated'):
            quiz_raw = central_agent.generate_quiz(quiz_data['topic'], quiz_data['grade'])
            questions = central_agent.parse_quiz(quiz_raw)
            
            if not questions:
                flash('Failed to generate quiz questions', 'error')
                return redirect(url_for('user_training'))
                
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
        return redirect(url_for('user_training'))

def submit_answer():
    if 'quiz' not in session:
        return redirect(url_for('user_training'))
    
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

def results():
    if 'quiz' not in session:
        return redirect(url_for('user_training'))
    
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
        return redirect(url_for('user_training'))

def study_plan():
    if 'quiz' not in session:
        return redirect(url_for('user_training'))
    
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

