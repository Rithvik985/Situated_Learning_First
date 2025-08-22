#!/usr/bin/env python3
"""
Assignment Evaluator using LLM and JSON Rubric
Evaluates student submissions against structured rubrics using OpenAI-compatible LLMs
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI
import argparse
import logging
from dotenv import load_dotenv   # 👈 add this

load_dotenv()

# model = os.getenv("VLLM_MODEL_FOR_ANALYSIS")
# url = os.getenv("VLLM_URL_FOR_ANALYSIS")
model = os.getenv("VLLM_MODEL_FOR_ANALYSIS", "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4")
url = os.getenv("VLLM_URL_FOR_ANALYSIS", "http://localhost:9091/v1/chat/completions")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Data class to store evaluation results for each question"""
    category: str
    question: str
    answer: str
    score: int
    reasoning: Optional[str] = None

class AssignmentEvaluator:
    """Main class for evaluating assignments using LLM and rubric"""
    
    # def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
    #     """
    #     Initialize the evaluator
        
    #     Args:
    #         api_key: OpenAI API key (or compatible)
    #         base_url: Base URL for the API (default OpenAI, can be changed for other providers)
    #         model: Model name to use for evaluation
    #     """
    #     self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    #     if not self.api_key:
    #         raise ValueError("API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
    #     self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    #     self.model = model
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url or os.getenv("VLLM_URL_FOR_ANALYSIS")
        self.model = model or os.getenv("VLLM_MODEL_FOR_ANALYSIS")
        self.timeout = timeout

        # ✅ Initialize OpenAI client (works with vLLM because it's OpenAI-compatible)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY")  # vLLM doesn’t check API keys
        )
        
    def load_rubric(self, rubric_path: str) -> Dict:
        """Load rubric from JSON file"""
        try:
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric = json.load(f)
            logger.info(f"Loaded rubric: {rubric.get('rubric_name', 'Unknown')}")
            return rubric
        except FileNotFoundError:
            raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rubric file: {e}")
    
    def load_submission(self, submission_path: str) -> str:
        """Load student submission from file"""
        try:
            with open(submission_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded submission from: {submission_path}")
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Submission file not found: {submission_path}")
    
    def create_evaluation_prompt(self, assignment_description: str, submission: str, question: str) -> str:
        """Create a prompt for LLM evaluation"""
        prompt = f"""You are an expert academic evaluator. Your task is to evaluate a student submission against a specific rubric question.

ASSIGNMENT DESCRIPTION:
{assignment_description}

STUDENT SUBMISSION:
{submission}

EVALUATION QUESTION:
{question}

INSTRUCTIONS:
1. Carefully read the assignment description and student submission
2. Evaluate whether the submission meets the criteria specified in the question
3. Base your evaluation strictly on the evidence present in the submission
4. If the information is not clearly present or insufficient, respond with "NO"

RESPONSE FORMAT:
Provide your response in exactly this format:
ANSWER: [YES/NO]
JUSTIFICATION: [10 words or less explaining your decision]

Example:
ANSWER: YES
JUSTIFICATION: Energy audit report clearly present in section 3."""

        return prompt
    
    def evaluate_question(self, assignment_description: str, submission: str, question: str, max_retries: int = 3) -> Tuple[str, int, str]:
        """
        Evaluate a single question using the LLM
        
        Returns:
            Tuple of (answer, score, justification) where answer is "YES"/"NO", score is 1/0, and justification is explanation
        """
        prompt = self.create_evaluation_prompt(assignment_description, submission, question)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise academic evaluator. Follow the exact format: ANSWER: [YES/NO] then JUSTIFICATION: [10 words or less]."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                answer, justification = self.parse_llm_response(response_text)
                
                if answer in ["YES", "NO"]:
                    score = 1 if answer == "YES" else 0
                    logger.debug(f"Question evaluated: {answer} (Score: {score}) - {justification}")
                    return answer, score, justification
                else:
                    logger.warning(f"Invalid response from LLM: {response_text}. Retrying...")
                    
            except Exception as e:
                logger.error(f"Error during evaluation (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to evaluate question after {max_retries} attempts")
                    return "NO", 0, "Evaluation failed due to technical error"
                time.sleep(1)  # Brief pause before retry
        
        return "NO", 0, "Evaluation failed after multiple attempts"
    
    def parse_llm_response(self, response_text: str) -> Tuple[str, str]:
        """
        Parse LLM response to extract answer and justification
        
        Returns:
            Tuple of (answer, justification)
        """
        try:
            lines = response_text.split('\n')
            answer = "NO"
            justification = "No justification provided"
            
            for line in lines:
                line = line.strip()
                if line.startswith("ANSWER:"):
                    answer = line.split(":", 1)[1].strip().upper()
                elif line.startswith("JUSTIFICATION:"):
                    justification = line.split(":", 1)[1].strip()
            
            # Fallback parsing if format not followed exactly
            if answer not in ["YES", "NO"]:
                # Try to find YES or NO anywhere in the response
                response_upper = response_text.upper()
                if "YES" in response_upper and "NO" not in response_upper:
                    answer = "YES"
                elif "NO" in response_upper and "YES" not in response_upper:
                    answer = "NO"
                else:
                    answer = "NO"  # Default to NO if unclear
                
                # Use entire response as justification if no proper format
                justification = response_text[:50] + ("..." if len(response_text) > 50 else "")
            
            # Truncate justification if too long
            if len(justification) > 80:  # Allow some flexibility beyond 10 words
                justification = justification[:77] + "..."
            
            return answer, justification
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return "NO", "Error parsing response"
    
    def evaluate_submission(self, assignment_description: str, submission: str, rubric: Dict) -> List[EvaluationResult]:
        """
        Evaluate entire submission against rubric
        
        Returns:
            List of EvaluationResult objects
        """
        results = []
        total_questions = sum(len(category['questions']) for category in rubric['rubrics'])
        
        logger.info(f"Starting evaluation of {total_questions} questions...")
        
        question_count = 0
        for category in rubric['rubrics']:
            category_name = category['category']
            logger.info(f"Evaluating category: {category_name}")
            
            for question in category['questions']:
                question_count += 1
                logger.info(f"Question {question_count}/{total_questions}: {question[:50]}...")
                
                answer, score, justification = self.evaluate_question(assignment_description, submission, question)
                
                result = EvaluationResult(
                    category=category_name,
                    question=question,
                    answer=answer,
                    score=score,
                    reasoning=justification
                )
                results.append(result)
                
                # Brief pause to avoid rate limiting
                time.sleep(0.5)
        
        return results
    
    def generate_report(self, results: List[EvaluationResult], rubric: Dict, submission_path: str) -> Dict:
        """Generate comprehensive evaluation report"""
        total_score = sum(result.score for result in results)
        total_questions = len(results)
        percentage = (total_score / total_questions) * 100 if total_questions > 0 else 0
        
        # Group results by category
        category_scores = {}
        for result in results:
            if result.category not in category_scores:
                category_scores[result.category] = {'correct': 0, 'total': 0}
            category_scores[result.category]['total'] += 1
            category_scores[result.category]['correct'] += result.score
        
        # Create detailed report
        report = {
            'rubric_name': rubric.get('rubric_name', 'Unknown'),
            'doc_type': rubric.get('doc_type', 'Unknown'),
            'submission_file': submission_path,
            'evaluation_summary': {
                'total_score': total_score,
                'total_questions': total_questions,
                'percentage': round(percentage, 2)
            },
            'category_breakdown': {},
            'detailed_results': []
        }
        
        # Add category breakdown
        for category, scores in category_scores.items():
            cat_percentage = (scores['correct'] / scores['total']) * 100
            report['category_breakdown'][category] = {
                'score': scores['correct'],
                'total': scores['total'],
                'percentage': round(cat_percentage, 2)
            }
        
        # Add detailed results
        for result in results:
            report['detailed_results'].append({
                'category': result.category,
                'question': result.question,
                'answer': result.answer,
                'score': result.score,
                'justification': result.reasoning or "No justification provided"
            })
        
        return report
    
    def save_report(self, report: Dict, output_path: str):
        """Save evaluation report to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def print_summary(self, report: Dict):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("ASSIGNMENT EVALUATION SUMMARY")
        print("="*60)
        print(f"Rubric: {report['rubric_name']}")
        print(f"Document Type: {report['doc_type']}")
        print(f"Submission: {report['submission_file']}")
        print("-"*60)
        
        summary = report['evaluation_summary']
        print(f"Overall Score: {summary['total_score']}/{summary['total_questions']} ({summary['percentage']:.1f}%)")
        print("-"*60)
        
        print("Category Breakdown:")
        for category, scores in report['category_breakdown'].items():
            print(f"  {category}: {scores['score']}/{scores['total']} ({scores['percentage']:.1f}%)")
        
        print("-"*60)
        print("Detailed Results:")
        for result in report['detailed_results']:
            status = "✓" if result['answer'] == "YES" else "✗"
            print(f"  {status} [{result['category']}] {result['question'][:50]}...")
            print(f"    Justification: {result['justification']}")
            print()
        
        print("="*60)

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate assignments using LLM and rubric")
#     parser.add_argument("assignment", help="Path to assignment description file")
#     parser.add_argument("submission", help="Path to student submission file")
#     parser.add_argument("rubric", help="Path to rubric JSON file")
#     parser.add_argument("--output", "-o", help="Output path for evaluation report (JSON)")
#     args = parser.parse_args()
    
#     try:
#         # Initialize evaluator directly from .env
#         evaluator = AssignmentEvaluator()
        
#         # Load files
#         assignment_description = evaluator.load_submission(args.assignment)
#         submission = evaluator.load_submission(args.submission)
#         rubric = evaluator.load_rubric(args.rubric)
        
#         # Evaluate submission
#         results = evaluator.evaluate_submission(assignment_description, submission, rubric)
        
#         # Generate report
#         report = evaluator.generate_report(results, rubric, args.submission)
        
#         # Print summary
#         evaluator.print_summary(report)
        
#         # Save report if output path specified
#         if args.output:
#             evaluator.save_report(report, args.output)
        
#     except Exception as e:
#         logger.error(f"Evaluation failed: {e}")
#         return 1
    
#     return 0


# if __name__ == "__main__":
#     exit(main())