import json
import httpx
from langchain.llms.base import LLM
from typing import Optional
import asyncio
import os
from dotenv import load_dotenv
from Agents.vision_agents import *
import fitz
from io import BytesIO
from docx import Document
from docx.parts.image import ImagePart
from PIL import Image
from typing import Dict, Tuple
from typing import List
from fastapi import UploadFile
import re
from InferenceEngine.inference_engines import *
# Load environment variables from .env file
load_dotenv()

# Access the environment variables
ollama_url = os.getenv("VLLM_URL_FOR_ANALYSIS")
ollama_model = os.getenv("VLLM_MODEL_FOR_ANALYSIS")
verba_url = os.getenv("VERBA_URL")

print(ollama_url)

# Define the OllamaLLM class with flexibility to input system and user prompts
class OllamaLLM(LLM):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt  # Set system prompt during initialization

    async def _invoke_llm(self, user_prompt: str) -> str:
        # Use the global ollama_url and the system prompt from initialization
        return await invoke_llm(self.system_prompt, user_prompt)

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        # Call the asynchronous invoke_llm in a synchronous manner
        response = asyncio.run(self._invoke_llm(prompt))
        return response["answer"]

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"


def resize_image(image_bytes: bytes, max_size: int = 800, min_size: int = 70) -> bytes:
    """
    Resize an image to ensure dimensions are between min_size and max_size while maintaining aspect ratio.
    
    Args:
        image_bytes: Original image bytes
        max_size: Maximum allowed size for any dimension
        min_size: Minimum allowed size for any dimension

    Returns:
        Resized image bytes
    """
    with Image.open(BytesIO(image_bytes)) as img:
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = orig_width / orig_height

        # Check if image needs to be resized up or down
        needs_upscaling = orig_width < min_size or orig_height < min_size
        needs_downscaling = orig_width > max_size or orig_height > max_size

        if needs_upscaling:
            # If width is smaller than minimum, scale up maintaining aspect ratio
            if orig_width < min_size:
                new_width = min_size
                new_height = int(new_width / aspect_ratio)
                # If height is still too small, scale based on height instead
                if new_height < min_size:
                    new_height = min_size
                    new_width = int(new_height * aspect_ratio)
            else:
                new_height = min_size
                new_width = int(new_height * aspect_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        elif needs_downscaling:
            # Use thumbnail for downscaling as it preserves aspect ratio
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Save the resized image
        output = BytesIO()
        img.save(output, format=img.format or 'PNG')  # Use PNG as fallback format
        return output.getvalue()

async def process_images_in_batch(
    images_data: List[Tuple[int, bytes]],
    batch_size: int = 5
) -> Dict[int, str]:
    """
    Process images in batches, resizing each image and sending them concurrently.
    Includes additional error handling and validation.

    Args:
        images_data: List of tuples containing (page_or_image_number, image_bytes)
        batch_size: Number of images to process in each batch

    Returns:
        Dictionary mapping page/image number to analysis result
    """
    ordered_results = {}

    for i in range(0, len(images_data), batch_size):
        batch = images_data[i:i + batch_size]

        try:
            # Resize images in the batch with minimum size requirement
            resized_batch = []
            for page_num, img_bytes in batch:
                try:
                    resized_img = resize_image(img_bytes, max_size=800, min_size=70)
                    resized_batch.append((page_num, resized_img))
                except Exception as e:
                    logger.error(f"Failed to resize image at page {page_num}: {e}")
                    continue

            # Skip batch if no images were successfully resized
            if not resized_batch:
                continue

            # Create async tasks for image analysis
            batch_tasks = [
                asyncio.create_task(analyze_image(img_bytes))
                for _, img_bytes in resized_batch
            ]

            # Run all tasks in the current batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results
            for (page_num, _), result in zip(resized_batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to analyze image at page {page_num}: {result}")
                    continue
                
                if isinstance(result, dict) and 'response' in result:
                    analysis_result = result['response'].strip()
                    if analysis_result:
                        ordered_results[page_num] = analysis_result

        except Exception as e:
            logger.error(f"Failed to process batch starting at index {i}: {e}")
            continue

    return dict(sorted(ordered_results.items()))


async def process_pdf(pdf_file: UploadFile) -> Dict[str, str]:
    """
    Process PDF file extracting text and images while preserving their original sequence.
    
    Args:
        pdf_file: Uploaded PDF file
    
    Returns:
        Dictionary with extracted text and image analyses in original sequence
    """
    pdf_bytes = await pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Use a list to maintain order instead of OrderedDict
    final_elements = []
    images_data = []

    # Start image analysis from page 7
    image_analysis_start_page = 6  # Pages are zero-indexed, so page 7 is index 6

    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Extract text using custom method for better block extraction
        page_text = extract_and_clean_text_from_page(page)
        if page_text:
            final_elements.append((page_num + 1, 'text', page_text))
        
        # Extract images from page, only analyze images starting from page 7
        if page_num >= image_analysis_start_page:
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images_data.append((page_num + 1, image_bytes))
                except Exception as e:
                    logger.error(f"Failed to extract image on page {page_num + 1}: {e}")

    # Process images in batches
    image_analyses = await process_images_in_batch(images_data) if images_data else {}
    
    # Insert image analyses into the final_elements list in their original positions
    for page_num, analysis in image_analyses.items():
        # Find the index where we want to insert the image analysis
        insert_index = next(
            (i for i, (p, type_, _) in enumerate(final_elements) 
             if p == page_num and type_ == 'text'), 
            len(final_elements)
        )
        
        # Insert image analysis right after the corresponding text
        final_elements.insert(insert_index + 1, (page_num, 'image', analysis))

    doc.close()
    
    # Combine text and image analyses in order
    combined_text = []
    for page_num, content_type, content in final_elements:
        if content_type == 'text':
            combined_text.append(content)
        else:  # image
            combined_text.append(f"\n[Image Analysis on Page {page_num}]: {content}")

    return {"text_and_image_analysis": "\n".join(combined_text).strip()}


async def process_docx(docx_file: UploadFile):
    """
    Process a DOCX file with batch image processing.
    """
    docx_bytes = await docx_file.read()
    docx_stream = BytesIO(docx_bytes)
    document = Document(docx_stream)
    final_text = ""

    # Process text
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            cleaned_text = re.sub(r'\s+', ' ', text)
            final_text += f" {cleaned_text}"

    # Prepare images for batch processing
    images_data = []
    for idx, rel in enumerate(document.part.rels.values()):
        if isinstance(rel.target_part, ImagePart):
            try:
                images_data.append((idx, rel.target_part.blob))
            except Exception as e:
                logger.error(f"Failed to extract DOCX image {idx}: {e}")

    # Process images in batches
    if images_data:
        analysis_results = await process_images_in_batch(images_data)

        # Add results to final text
        for idx, analysis_result in sorted(analysis_results.items()):
            final_text += f"\n\nImage Analysis (Image {idx + 1}): {analysis_result}"
            
    cleaned_text = clean_text(final_text)
    return {"text_and_image_analysis": cleaned_text.strip()}


def extract_and_clean_text_from_page(page) -> str:
    """
    Extract and clean text from a PDF page using PyMuPDF.
    
    Args:
        page: PyMuPDF page object
    
    Returns:
        Cleaned text string
    """
    text_blocks = []
    blocks = page.get_text("blocks")
    for block in blocks:
        if isinstance(block[4], str) and block[4].strip():
            cleaned_block = ' '.join(block[4].split())
            if cleaned_block:
                text_blocks.append(cleaned_block)

    combined_text = ' '.join(text_blocks)
    return clean_text(combined_text)

def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing unnecessary elements.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text string
    """
    import re
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Chapter\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\b(?!\s*[a-zA-Z])', '', text)
    text = re.sub(r'[\r\n\t\f]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()



async def  call_spanda_retrieve(payload):

    url = f"{verba_url}/api/query"
    request_data = payload
   
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=request_data)
        # Check if the response was successful
        if response.status_code == 200:
            is_response_relevant = response.json()
            print("is_response_relevant",is_response_relevant) 

            return is_response_relevant
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        

async def response_relevance_filter(query: str, response: str) -> str:
    evaluate_system_prompt = """You are given a query and a response. Determine if the response is relevant, irrelevant, or highly irrelevant to the query. Only respond with "Relevant", "Irrelevant", or "Highly Irrelevant"."""

    evaluate_user_prompt = f"""
        Query: {query}

        Content: {response}
    """
    
    is_response_relevant_dict = await invoke_llm(
        system_prompt=evaluate_system_prompt,
        user_prompt=evaluate_user_prompt,
        model_type= ModelType.ANALYSIS
    )
    is_response_relevant = is_response_relevant_dict["answer"]
    if is_response_relevant.lower() == 'highly irrelevant':
        return "Given that the answer that I am able to retrieve with the information I have seems to be highly irrelevant to the query, I abstain from providing a response. I am sorry for not being helpful." # Returns an empty coroutine
    elif is_response_relevant.lower() == 'irrelevant':
        return "The answer I am able to retrieve with the information I have seems to be irrelevant to the query. Nevertheless, I will provide you with the response in the hope that it will be valuable. Apologies in advance if it turns out to be of no value: " + response
    return response


async def context_relevance_filter(query: str, context: str) -> str:
    evaluate_system_prompt = (
        """You are an AI responsible for assessing whether the provided content is relevant to a specific query. Carefully analyze the content and determine if it directly addresses or provides pertinent information related to the query. Only respond with "YES" if the content is relevant, or "NO" if it is not. Do not provide any explanations, scores, or additional text—just a single word: "YES" or "NO"."""
    )
    evaluate_user_prompt = (
        f"""
        Content: {context}

        Query: {query}

        You are an AI responsible for assessing whether the provided content is able to answer the query. Carefully analyze the content and determine if it directly addresses or provides pertinent information related to the query. Only respond with "YES" if the content is relevant, or "NO" if it is not. Do not provide any explanations, scores, or additional text—just a single word: "YES" or "NO".
        """
    )

    is_context_relevant_dict = await invoke_llm(
        system_prompt=evaluate_system_prompt,
        user_prompt=evaluate_user_prompt,
        model_type= ModelType.ANALYSIS
    )
    is_context_relevant = is_context_relevant_dict["answer"]
    if is_context_relevant.lower() == 'no':
        return " "  # Returns an empty coroutine
    return context



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
url = os.getenv("VLLM_OPENAIURL_FOR_ANALYSIS", "http://localhost:9091/v1/")


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
        # Debug: Check what environment variables are set
        env_vars = {k: v for k, v in os.environ.items() if 'VLLM' in k or 'OPENAI' in k}
        logger.info(f"Relevant environment variables: {env_vars}")
        
        # Get values from environment variables or use defaults
        env_base_url = os.getenv("VLLM_OPENAIURL_FOR_ANALYSIS")
        env_model = os.getenv("VLLM_MODEL_FOR_ANALYSIS")
        
        self.base_url = base_url or env_base_url or "http://localhost:9091/v1"
        self.model = model or env_model or "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
        self.timeout = timeout

        logger.info(f"Raw base URL: {self.base_url}")
        
        # ✅ Clean up the base URL
        if self.base_url:
            # Remove trailing slash
            self.base_url = self.base_url.rstrip('/')
            # Remove /chat/completions if it's included
            if '/chat/completions' in self.base_url.lower():
                self.base_url = self.base_url.lower().replace('/chat/completions', '')
        
        logger.info(f"Cleaned base URL: {self.base_url}")

        # ✅ Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
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