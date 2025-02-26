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
            
        session['first_quiz'] = {
            'grade': grade,
            'subject': subject,
            'topic': topic,
            'questions': [],
            'answers': [],
            'start_time': None,
            'current_q': 0,
            'generated': False,
        }
        return redirect(url_for('first_quiz'))
    except Exception as e:
        logger.error(f"Error starting learning session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('user_training'))

def first_quiz():
    if 'first_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['first_quiz']
        
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
            session['first_quiz'] = quiz_data

        questions = quiz_data.get('questions', [])
        current_page = quiz_data['current_q'] // 5
        start_idx = current_page * 5
        end_idx = min((current_page + 1) * 5, len(questions))
        current_questions = questions[start_idx:end_idx]
        
        return render_template(
            'first_quiz.html',
            questions=current_questions,
            current_page=current_page,
            total_pages=(len(questions) + 4) // 5,
            start_idx=start_idx
        )
    except Exception as e:
        logger.error(f"Error displaying first quiz: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('user_training'))

def submit_first_quiz():
    if 'first_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['first_quiz']
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
        session['first_quiz'] = quiz_data

        current_page = quiz_data['current_q'] // 5
        total_questions = len(quiz_data['questions'])
        total_pages = (total_questions + 4) // 5

        if current_page >= total_pages - 1:
            return redirect(url_for('first_results'))
        else:
            quiz_data['current_q'] = (current_page + 1) * 5
            session['first_quiz'] = quiz_data
            return redirect(url_for('first_quiz'))
    except Exception as e:
        logger.error(f"Error submitting first quiz answer: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('first_quiz'))

def first_results():
    if 'first_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['first_quiz']
        correct = sum(1 for ans, q in zip(quiz_data['answers'], quiz_data['questions']) 
                     if ans and q and ans.strip() == q["answer"].strip())
        
        total_time = time.time() - quiz_data['start_time']
        proficiency = central_agent.assess_proficiency(
            correct, 
            len(quiz_data['questions']), 
            total_time
        )
        
        session['initial_proficiency'] = proficiency
        session['initial_score'] = (correct / len(quiz_data['questions'])) * 100
        return render_template(
            'results.html',
            correct=correct,
            total=len(quiz_data['questions']),
            time_taken=int(total_time),
            proficiency=proficiency,
            next_url=url_for('study_plan')
        )
    except Exception as e:
        logger.error(f"Error displaying first results: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('user_training'))

def study_plan():
    if 'first_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['first_quiz']
        subject = quiz_data['subject']
        agent = central_agent.math_agent if subject == "Math" else central_agent.science_agent
        
        # Process study plan to remove markdown and code block artifacts
        def clean_content(content):
            # Remove markdown code block markers
            content = content.replace('```html', '').replace('```', '')
            # Remove section markers
            content = content.replace('---', '')
            # Remove any remaining markdown formatting
            content = content.replace('**', '')
            # Split into lines
            lines = content.split('\n')
            # Clean each line while preserving HTML
            cleaned_lines = []
            for line in lines:
                if line.strip():
                    # Preserve indentation
                    indent = len(line) - len(line.lstrip())
                    cleaned_line = ' ' * indent + line.strip()
                    cleaned_lines.append(cleaned_line)
            # Join back preserving line breaks
            return '\n'.join(cleaned_lines)

        # Generate study plan content
        raw_study_plan = agent.generate_content(f"""
            Create a detailed study plan for {quiz_data['topic']} for grade {quiz_data['grade']} students.

            IMPORTANT: Return ONLY the following HTML structure without any additional text or markdown:

            <div class="topic-section">
                <div class="topic-title">Foundations (45 minutes)</div>
                <div class="learning-objectives">
                    <strong>Learning Goals:</strong>
                    <ul class="study-list">
                        <li>Master the fundamental concepts of {quiz_data['topic']}</li>
                        <li>Build problem-solving confidence</li>
                    </ul>
                </div>
                <div class="subtopic-title">Key Concepts</div>
                <ul class="study-list">
                    <li>Core principle explanations</li>
                    <li>Important formulas and rules</li>
                </ul>
                <div class="practice-exercises">
                    <strong>Practice Activities:</strong>
                    <ul class="study-list">
                        <li>Specific practice problems</li>
                        <li>Hands-on exercises</li>
                    </ul>
                </div>
                <div class="milestone">
                    <strong>Progress Check:</strong>
                    List specific skills mastered in this section
                </div>
            </div>

            Create 4 sections with proper HTML indentation:
            1. Foundations (45 minutes)
            2. Core Concepts (60 minutes)
            3. Advanced Applications (45 minutes)
            4. Mastery Review (30 minutes)

            Each section should:
            - Have clear, measurable learning goals
            - Include specific examples related to {quiz_data['topic']}
            - List detailed practice exercises
            - Define progress milestones
        """)

        study_plan = clean_content(raw_study_plan)
        
        # Generate guide content with proper structure
        raw_guide = agent.generate_content(f"""
            Create a detailed guide for {quiz_data['topic']} in {subject} 
            for grade {quiz_data['grade']} students.

            IMPORTANT: Return ONLY the following HTML structure without any additional text or markdown:

            <div class="topic-section">
                <div class="topic-title">Understanding {quiz_data['topic']}</div>
                
                <div class="key-concept">
                    <strong>Key Points:</strong>
                    <ul class="study-list">
                        <li>Fundamental principle 1 with clear explanation</li>
                        <li>Important rule or formula with context</li>
                        <li>Common applications and uses</li>
                    </ul>
                </div>

                <div class="subtopic-title">Step-by-Step Understanding</div>
                <p>Clear, detailed explanation of how this concept works and why it's important.</p>
                
                <div class="example-box">
                    <strong>Example Problem:</strong>
                    <p>Step 1: Initial setup and what to look for</p>
                    <p>Step 2: Key procedure with explanation</p>
                    <p>Step 3: Final solution and verification</p>
                </div>
            </div>

            Create 3 sections:
            1. Basic Concepts
            2. Problem-Solving Strategies
            3. Advanced Applications

            Each section should:
            - Explain concepts clearly for grade {quiz_data['grade']} level
            - Include relevant examples
            - Highlight common mistakes to avoid
            - Show practical applications
        """)

        guide = clean_content(raw_guide)
        
        # Initialize final quiz with same parameters
        session['final_quiz'] = {
            'grade': quiz_data['grade'],
            'subject': quiz_data['subject'],
            'topic': quiz_data['topic'],
            'questions': [],
            'answers': [],
            'start_time': None,
            'current_q': 0,
            'generated': False,
        }
        
        return render_template(
            'study_plan.html',
            study_plan=study_plan,
            guide=guide,
            next_url=url_for('final_quiz')
        )
    except Exception as e:
        logger.error(f"Error generating study plan: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('first_results'))

def final_quiz():
    if 'final_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['final_quiz']
        
        if not quiz_data.get('generated'):
            quiz_raw = central_agent.generate_quiz(quiz_data['topic'], quiz_data['grade'])
            questions = central_agent.parse_quiz(quiz_raw)
            
            if not questions:
                flash('Failed to generate quiz questions', 'error')
                return redirect(url_for('study_plan'))
                
            quiz_data.update({
                "questions": questions,
                "start_time": time.time(),
                "current_q": 0,
                "generated": True
            })
            session['final_quiz'] = quiz_data

        questions = quiz_data.get('questions', [])
        current_page = quiz_data['current_q'] // 5
        start_idx = current_page * 5
        end_idx = min((current_page + 1) * 5, len(questions))
        current_questions = questions[start_idx:end_idx]
        
        return render_template(
            'final_quiz.html',
            questions=current_questions,
            current_page=current_page,
            total_pages=(len(questions) + 4) // 5,
            start_idx=start_idx
        )
    except Exception as e:
        logger.error(f"Error displaying final quiz: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('study_plan'))

def submit_final_quiz():
    if 'final_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['final_quiz']
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
        session['final_quiz'] = quiz_data

        current_page = quiz_data['current_q'] // 5
        total_questions = len(quiz_data['questions'])
        total_pages = (total_questions + 4) // 5

        if current_page >= total_pages - 1:
            return redirect(url_for('final_results'))
        else:
            quiz_data['current_q'] = (current_page + 1) * 5
            session['final_quiz'] = quiz_data
            return redirect(url_for('final_quiz'))
    except Exception as e:
        logger.error(f"Error submitting final quiz answer: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('final_quiz'))

def final_results():
    if 'final_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        quiz_data = session['final_quiz']
        correct = sum(1 for ans, q in zip(quiz_data['answers'], quiz_data['questions']) 
                     if ans and q and ans.strip() == q["answer"].strip())
        
        total_time = time.time() - quiz_data['start_time']
        proficiency = central_agent.assess_proficiency(
            correct, 
            len(quiz_data['questions']), 
            total_time
        )
        
        session['final_proficiency'] = proficiency
        session['final_score'] = (correct / len(quiz_data['questions'])) * 100
        return redirect(url_for('comparison'))
    except Exception as e:
        logger.error(f"Error displaying final results: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('final_quiz'))

def comparison():
    if 'first_quiz' not in session or 'final_quiz' not in session:
        return redirect(url_for('user_training'))
    
    try:
        # Calculate initial quiz stats
        first_quiz = session['first_quiz']
        initial_correct = sum(1 for ans, q in zip(first_quiz['answers'], first_quiz['questions']) 
                            if ans and q and ans.strip() == q["answer"].strip())
        initial_total = len(first_quiz['questions'])
        initial_score = (initial_correct / initial_total) * 100

        # Calculate final quiz stats
        final_quiz = session['final_quiz']
        final_correct = sum(1 for ans, q in zip(final_quiz['answers'], final_quiz['questions']) 
                          if ans and q and ans.strip() == q["answer"].strip())
        final_total = len(final_quiz['questions'])
        final_score = (final_correct / final_total) * 100

        improvement = final_score - initial_score
        
        # Generate improvement-based recommendations
        recommendations = []
        if improvement < 0:
            recommendations.append({
                'topic': 'Core Concepts Review',
                'advice': 'Consider reviewing the fundamental concepts again. Focus on understanding the basics before moving to advanced topics.'
            })
        elif improvement < 10:
            recommendations.append({
                'topic': 'Practice More',
                'advice': 'You\'re making progress but need more practice. Try additional exercises focusing on areas where you scored lower.'
            })
        else:
            recommendations.append({
                'topic': 'Advanced Learning',
                'advice': 'Great improvement! You\'re ready for more advanced topics. Consider exploring related concepts to broaden your knowledge.'
            })
        
        return render_template(
            'comparison.html',
            initial_score=round(initial_score, 1),
            final_score=round(final_score, 1),
            improvement=round(improvement, 1),
            initial_correct=initial_correct,
            initial_total=initial_total,
            final_correct=final_correct,
            final_total=final_total,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error displaying comparison: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('user_training'))
