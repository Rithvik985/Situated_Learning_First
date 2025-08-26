from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
from src.components.rubric_generator import *
from dotenv import load_dotenv   # 👈 add this

load_dotenv()

# model = os.getenv("VLLM_MODEL_FOR_ANALYSIS")
# url = os.getenv("VLLM_URL_FOR_ANALYSIS")
model = os.getenv("VLLM_MODEL_FOR_ANALYSIS", "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4")
url = os.getenv("VLLM_URL_FOR_ANALYSIS", "http://localhost:9091/v1/chat/completions")

app = FastAPI(title="Rubric Generator API")

print("Using URL:", url)
print("Using Model:", model)

client = GenericLLM(
    base_url=url,  # or your API endpoint
    model=model    # or whatever model you are using
)

class RubricRequest(BaseModel):
    text: str
    doc_type: str

@app.post("/generate_rubric")
def generate_rubric(request: RubricRequest):
    try:
        result = generate_objective_rubric(request.text, request.doc_type, client, model=model)
        return {"status": "success", "rubric": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ✅ Run directly with python main.py
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=6022, reload=True)
