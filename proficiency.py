from config_agent import logger
import time
import google.generativeai as genai
from config_agent import CENTRAL_API_KEY

# Configure Gemini
try:
    genai.configure(api_key=CENTRAL_API_KEY)
    proficiency_model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")
    raise

def assess_proficiency(correct_count, total_questions, total_time):
    """
    Assess student proficiency using Gemini AI.
    Returns: 'high', 'intermediate', or 'low'
    """
    try:
        if not all(isinstance(x, (int, float)) for x in [correct_count, total_questions, total_time]):
            raise ValueError("Invalid input types for proficiency assessment")
            
        if total_questions == 0:
            return "unknown"
            
        # Calculate metrics
        percentage = (correct_count / total_questions) * 100
        avg_time_per_question = total_time / total_questions

        # Create detailed prompt for Gemini
        prompt = f"""
        You are a proficiency assessment system. Given a student's quiz performance, determine their proficiency level.

        Student Performance Data:
        - Correct answers: {correct_count} out of {total_questions} questions
        - Success rate: {percentage:.1f}%
        - Average time per question: {avg_time_per_question:.1f} seconds
        - Total quiz duration: {total_time:.1f} seconds

        Proficiency Level Guidelines:
        - HIGH: Strong understanding (≥80% correct) with quick responses
        - INTERMEDIATE: Good understanding (50-79% correct) or moderate pace
        - LOW: Needs improvement (<50% correct) or very slow responses

        Based on these metrics, respond with exactly one word from: high, intermediate, or low
        No other text or explanation - just the proficiency level word.
        """

        try:
            # Make multiple attempts if needed
            for attempt in range(3):
                response = proficiency_model.generate_content(prompt)
                time.sleep(0.5)  # Rate limiting
                
                result = response.text.strip().lower()
                if result in ['high', 'intermediate', 'low']:
                    logger.info(f"Proficiency assessment: {result} (percentage: {percentage:.1f}%, avg time: {avg_time_per_question:.1f}s)")
                    return result
                
            logger.error("Gemini failed to provide valid response after 3 attempts")
            # Fallback to deterministic assessment
            if percentage >= 80 and avg_time_per_question <= 90:
                return "high"
            elif percentage < 50 or avg_time_per_question >= 150:
                return "low"
            else:
                return "intermediate"
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Fallback to deterministic assessment
            if percentage >= 80 and avg_time_per_question <= 90:
                return "high"
            elif percentage < 50 or avg_time_per_question >= 150:
                return "low"
            else:
                return "intermediate"
            
    except Exception as e:
        logger.error(f"Error assessing proficiency: {str(e)}")
        return "unknown"
