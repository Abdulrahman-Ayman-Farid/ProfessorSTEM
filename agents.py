import time
import google.generativeai as genai
from config_agent import logger, CENTRAL_API_KEY, MATH_API_KEY, SCIENCE_API_KEY
from proficiency import assess_proficiency

# Configure Gemini clients
try:
    genai.configure(api_key=CENTRAL_API_KEY)
    central_model = genai.GenerativeModel('gemini-1.5-flash')

    genai.configure(api_key=MATH_API_KEY)
    math_model = genai.GenerativeModel('gemini-1.5-flash')

    genai.configure(api_key=SCIENCE_API_KEY)
    science_model = genai.GenerativeModel('gemini-1.5-flash')

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

    def generate_quiz(self, topic, grade):
        if not isinstance(topic, str) or not isinstance(grade, int):
            raise ValueError("Invalid topic or grade format")
        
        from flask import session
        quiz_data = session.get('first_quiz', {})
        
        # Use provided age and school level from session
        age = quiz_data.get('age', grade + 5)  # Fallback to calculated age if not in session
        school_level = quiz_data.get('school_level', 'Middle School')  # Fallback to middle school if not in session
            
        prompt = f"""
        Generate a 25-question MCQ Exam on {topic} for {school_level} students in grade {grade} (age: {age}).

        Consider the following factors when generating questions:
        1. Age-appropriate language and complexity (for {age}-year-olds)
        2. {school_level} curriculum standards and expectations
        3. Typical attention span and cognitive development at age {age}
        4. Real-world examples that {age}-year-old students can relate to

        Format each question exactly like:
        [Question X] What is the question text? | Option 1 | Option 2 | Option 3 | Option 4 | Correct Answer

        Example:
        [Question 1] What is 2+2? | 3 | 4 | 5 | 6 | 4

        Make questions progressively harder and include:
        1. Basic recall questions (33%) - Match {school_level} comprehension level
        2. Understanding/application questions (33%) - Use age-appropriate examples
        3. Analysis/problem-solving questions (34%) - Align with cognitive abilities at age {age}

        Ensure questions are:
        - Using vocabulary appropriate for {age}-year-olds
        - Referencing concepts familiar to {school_level} students
        - Structured at the right difficulty level for grade {grade}
        - Including relatable examples for this age group
        """
        # Use appropriate agent based on topic difficulty
        if grade <= 6:
            result = self.math_agent.generate_content(prompt)
        else:
            # For higher grades, use science agent which might be better at complex topics
            result = self.science_agent.generate_content(prompt)
            
        # Validate generated questions
        questions = self.parse_quiz(result)
        if not questions or len(questions) < 10:  # If we got too few questions
            logger.warning(f"Insufficient questions generated ({len(questions) if questions else 0}), retrying...")
            # Try with the other agent as backup
            result = (self.science_agent if grade <= 6 else self.math_agent).generate_content(prompt)
            
        return result

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
            
            # Validate parsed questions
            if questions:
                logger.info(f"Successfully parsed {len(questions)} questions")
                # Log first question as sample for verification
                if questions[0]:
                    logger.info(f"Sample question: {questions[0]['question']}")
            else:
                logger.warning("No questions were parsed from the generated text")
                
            return questions
        except Exception as e:
            logger.error(f"Error parsing quiz: {str(e)}")
            return []

    def assess_proficiency(self, correct_count, total_questions, total_time):
        result = assess_proficiency(correct_count, total_questions, total_time)
        logger.info(f"Proficiency assessment result: {result} ({correct_count}/{total_questions} correct, {total_time:.1f}s total)")
        return result
