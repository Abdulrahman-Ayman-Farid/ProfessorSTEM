import time
import google.generativeai as genai
from config_agent import logger, CENTRAL_API_KEY, MATH_API_KEY, SCIENCE_API_KEY
from proficiency import assess_proficiency

# Configure Gemini clients
try:
    genai.configure(api_key=CENTRAL_API_KEY)
    central_model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    genai.configure(api_key=MATH_API_KEY)
    math_model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    genai.configure(api_key=SCIENCE_API_KEY)
    science_model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

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

        Make sure each question is appropriate for grade {grade} level students.
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
