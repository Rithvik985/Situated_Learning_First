import os
from dotenv import load_dotenv
import aiohttp
import base64
import logging
import base64
from typing import List, Dict, Any
import logging

from InferenceEngine.inference_engines import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a logger instance
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
ollama_url = os.getenv("OLLAMA_URL")
ollama_model_for_image = os.getenv("OLLAMA_MODEL_FOR_IMAGE")

vllm_url_for_image = os.getenv("VLLM_URL_FOR_IMAGE")
vllm_model_for_image = os.getenv("VLLM_MODEL_FOR_IMAGE")


# choose agent
async def analyze_image(image_data: bytes) -> Dict[str, Any]:
    """
    Unified method to analyze images using either VLLM or Ollama based on available environment variables.
    VLLM takes precedence if both are configured.
    
    Args:
        image_data: Image bytes to analyze
        
    Returns:
        Dict containing the analysis response
    
    Raises:
        ValueError: If neither VLLM nor Ollama environment variables are configured
    """
    # Check VLLM configuration
    has_vllm = all([
        vllm_url_for_image,
        vllm_model_for_image
    ])
    
    # Check Ollama configuration
    has_ollama = all([
        ollama_url,
        ollama_model_for_image
    ])
    
    # If neither service is configured, raise error
    if not (has_vllm or has_ollama):
        raise ValueError(
            "No image analysis service configured. Please set either VLLM "
            "(VLLM_URL_FOR_IMAGE, VLLM_MODEL_FOR_IMAGE) or Ollama "
            "(OLLAMA_URL, OLLAMA_MODEL_FOR_IMAGE) environment variables."
        )
    
    try:
        # Prefer VLLM if available
        if has_vllm:
            logger.info("Using VLLM for image analysis")
            return await analyze_image_vllm(image_data)
        else:
            logger.info("Using Ollama for image analysis")
            return await analyze_image_ollama(image_data)
            
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return {"response": f"Failed to analyze image: {str(e)}"}


###############################################################################################################################################################
###############################################################################################################################################################
#########################################VLLM IMAGE AGENT###############################################################################################
###############################################################################################################################################################

async def analyze_image_vllm(
    image_data: bytes,
    model: str = vllm_model_for_image,
    base_url: str = vllm_url_for_image
) -> Dict[str, Any]:
    """
    Analyze image using predefined prompt
    
    Args:
        image_data: Image bytes
        model: Model identifier
        base_url: API base URL
    
    Returns:
        Dict containing the analysis response
    """

    image_agent_user_prompt = """
Analyze the following image and provide a report detailing the features present. Include a clear description of what is depicted in the image without any interpretation.
Provide detailed Analysis of the image without missing facts. Do not miss any textual or numbers so that it is a very comprehensive analysis. Do NOT solve for missing values, or modify. 
Your goal is to provide a description such that someone else can re-create the image.
    """
    
    return await generate_from_image(
        image_data=image_data,
        prompt=image_agent_user_prompt,
        model=model,
        base_url=base_url
    )

###############################################################################################################################################################
#########################################OLLAMA IMAGE AGENT####################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################


async def analyze_image_ollama(image_data: bytes):
    image_agent_user_prompt = """
Analyze the following image and provide a report detailing the features present. Include a clear description of what is depicted in the image without any interpretation.
Provide detailed Analysis of the image without missing facts. Do not miss any textual or numbers so that it is a very comprehensive analysis. Do NOT solve for missing values, or modify. 
Your goal is to provide a description such that someone else can re-create the image.
    """
    return await generate_from_image_ollama(image_data, image_agent_user_prompt)


###############################################################################################################################################################
#########################################HELPER FUNCTIONS######################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

async def generate_from_image_ollama(image_data: bytes, prompt: str):
    try:
        # Encode the binary image data to Base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        data = {
            "model": ollama_model_for_image,
            "prompt": prompt,
            "images": [encoded_image],
            "options": {
                "top_k": 1, 
                "top_p": 0, 
                "temperature": 0,
                "seed": 100
            },
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{ollama_url}/api/generate", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print("########################################")
                    print(result)
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Image analysis failed with status {response.status}: {error_text}")
                    return {"response": "Failed to analyze image"}
    except Exception as e:
        logger.error(f"Error in generate_from_image: {str(e)}")
        return {"response": "Failed to analyze image"}



async def generate_from_image(
    image_data: bytes,
    prompt: str,
    model: str = vllm_model_for_image,
    base_url: str = vllm_url_for_image
) -> Dict[str, Any]:
    """
    Generate response from image using custom API format
    
    Args:
        image_data: Image bytes
        prompt: The prompt text
        model: Model identifier
        base_url: API base URL
    
    Returns:
        Dict containing the response
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    try:
        response = await send_multimodal_chat_message(
            messages=messages,
            image_bytes=image_data,
            model=model,
            base_url=base_url
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_from_image: {str(e)}")
        return {"response": "Failed to analyze image"}


async def send_multimodal_chat_message(
    messages: List[Dict[str, Any]],
    image_bytes: bytes = None,
    model: str = "Qwen/Qwen2-VL-2B-Instruct-AWQ",
    base_url: str = "http://localhost:8002/v1/chat/completions"
) -> str:
    """
    Send a multi-modal chat message with optional image bytes to the API endpoint.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        image_bytes: Optional image data as bytes
        model: The model identifier to use
        base_url: Base URL of the API server
    
    Returns:
        The response text from the model
    """
    endpoint = f"{base_url}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if image_bytes:
        try:
            base64_image = await encode_bytes_to_base64(image_bytes)
            for msg in reversed(messages):
                if msg["role"] == "user":
                    msg["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    )
                    break
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.1,
        "top_k": 1,
        "seed": 42,
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed with status {response.status}: {error_text}")
                    raise aiohttp.ClientError(f"API request failed: {error_text}")
                    
    except Exception as e:
        logger.error(f"Error making API request: {str(e)}")
        raise


async def encode_bytes_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        Base64 encoded string of the image
    """
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise