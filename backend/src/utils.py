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