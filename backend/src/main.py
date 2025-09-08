import sys
import os
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import FastAPI, HTTPException, Request, Body, Query,Depends,APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from src.utils import *
from typing import Dict, List, Optional, Union, Any, Literal
from src.data_types_class import *
from fastapi.encoders import jsonable_encoder
from src.configs import *
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from database.init import init_db,init_minio_table
from sqlalchemy.orm import Session
from Question_Extractor.extractor import process_single_pdf_with_verification
from pydantic import BaseModel
from dotenv import load_dotenv
from database.connector import SessionLocal
from database.models import AssignmentContent,File  # adjust import path if different
from fastapi import Depends
import json
import logging
from backend.src.utils import AssignmentEvaluator

RUBRIC_API_URL = "http://localhost:6022/generate_rubric"

llm_url = os.getenv("VLLM_URL_FOR_ANALYSIS")
llm_model = os.getenv("VLLM_MODEL_FOR_ANALYSIS")
verba_url = os.getenv("VERBA_URL")

from database.models import Assignment
from database.deps import get_db
from database.connector import *
# Add this import to fix the ModelType issue
from InferenceEngine.inference_engines import ModelType
# Place this at top of main.py
from uuid import uuid4
session_store = {}

class CourseIDRequest(BaseModel):
    course_id:str

app = FastAPI(
    title="Situated Learning App",
    description="Backend Situated Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CancellationToken:
    def __init__(self):
        self.is_cancelled = False
        self.ws_closed = False

    def cancel(self):
        self.is_cancelled = True

    def mark_closed(self):
        self.ws_closed = True

def extract_assignment_text_from_pdf(path: str) -> str:
    try:
        print(f"Extracting assignment text from: {path}")
        output = process_single_pdf_with_verification(
            pdf_path=path,
            model=llm_model,
            base_url=llm_url,
        )
        return output.strip()
    except Exception as e:
        print(f"[ERROR] PDF extraction failed for {path}: {e}")
        return "[Failed to extract content]"


@app.on_event("startup")
async def startup_event():
    init_db()
    init_minio_table()

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running on port 8090!"}

@app.get("/llm_status")
async def llm_status():
    try:
        # Minimal sanity check prompt
        system_prompt = "You are a health check bot. Reply with 'OK'."
        user_prompt = "Health check."

        llm_response = await invoke_llm(system_prompt, user_prompt,model_type=ModelType.ANALYSIS)
        answer = llm_response.get("answer", "")

        if "ok" in answer.lower():
            return {"llm_live": True, "response": answer}
        else:
            return {"llm_live": False, "response": answer}

    except Exception as e:
        return {"llm_live": False, "error": str(e)}


# @app.get("/assignments/by_course/{course_id}")
# def get_assignments_by_course(course_id: str):
#     db: Session = SessionLocal()
#     try:
#         assignments = db.query(Assignment).filter(Assignment.course_id == course_id).all()
#         return [
#             {
#                 "id": a.id,
#                 "course_title": a.course_title,
#                 "instructor_name": a.instructor_name,
#                 "pdf_link": a.pdf_link,
#                 "topic": a.topic
#             } for a in assignments
#         ]
#     finally:
#         db.close()

@app.get("/courses/all")
def get_all_courses(db: Session = Depends(get_db)):
    """
    Get all unique course names from the File database table.
    
    Returns:
        List of unique course titles
    """
    try:
        # Query all distinct course titles
        courses = db.query(File.course_title).distinct().all()
        
        # Extract course titles from the result tuples
        course_list = [course[0] for course in courses if course[0] is not None]
        
        return {
            "courses": course_list,
            "count": len(course_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching courses: {str(e)}")
    
# @app.post("/start_assignment_session")
# def start_assignment_session(payload: CourseIDRequest):
#     course_id=payload.course_id
#     db: Session = SessionLocal()
#     try:
#         assignments = db.query(Assignment).filter(Assignment.course_id == course_id).all()

#         if not assignments:
#             raise HTTPException(status_code=404, detail="No assignments found for this course.")

#         # examples = [
#         #     {
#         #         "course_title": a.course_title,
#         #         "instructor": a.instructor_name,
#         #         "topic": a.topic,
#         #         "pdf_link": a.pdf_link
#         #     }
#         #     for a in assignments
#         # ]

#         examples = []
#         for a in assignments:
#             #full_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), a.pdf_link))
#             relative_pdf_path = os.path.normpath(a.pdf_link)  # ensures proper slashes
#             full_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_pdf_path))
  
#             assignment_text = extract_assignment_text_from_pdf(full_pdf_path)
            
#             examples.append({
#                 "course_title": a.course_title,
#                 "topic": a.topic,
#                 "content": assignment_text
#             })


#         session_id = str(uuid4())
#         session_store[session_id] = {
#             "course_id": course_id,
#             "examples": examples
#         }

#         return {
#             "message": f"Session started for course_id={course_id}",
#             "session_id": session_id,
#             "examples": examples  # <--- now included in response

#         }
#     finally:
#         db.close()



@app.get("/assignments/by_course_title/{course_title}")
def get_assignments_by_course_title(course_title: str, db: Session = Depends(get_db)):
    try:
        assignments = db.query(File).filter(File.course_title == course_title).all()
        if not assignments:
            raise HTTPException(status_code=404, detail="No assignments found for this course_title.")

        return [
            {
                "id": a.id,
                "course_title": a.course_title,
                "Content": a.content,
            } for a in assignments
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_from_course_title")
async def generate_from_course_title(
    course_title: str = Body(...),
    topic: str = Body(...),
    user_domain: str = Body(...),
    extra_instructions: Optional[str] = Body(default=""),
    db: Session = Depends(get_db)
):
    # --- Fetch assignments with exact match ---
    assignments = db.query(File).filter(File.course_title == course_title).all()
    if not assignments:
        raise HTTPException(status_code=404, detail="No assignments found for this course_title.")

    examples = []
    for a in assignments:

        examples.append({
            "course_title": a.course_title,
            "content": a.content
        })   


        examples_text = "\n\n".join([
            f"### Example Assignment\n"
            f"Course Title: {ex['course_title']}\n"
            f"Content:{ex['content']}\n"
            for ex in examples
        ])

    system_prompt = (
        "You are an AI assistant specialized in creating industry-relevant, practical assignments for working professionals. "
        "Your primary goal is to generate assignments that have immediate real-world application in the student's workplace and industry domain. "
        
        "Key Requirements:\n"
        "- Create assignments that solve actual industry problems students encounter in their work\n"
        "- Design tasks that produce deliverables students can use in their current role or company\n"
        "- Focus on practical skills and competencies that directly impact job performance\n"
        "- Incorporate industry-specific tools, methodologies, and best practices\n"
        "- Ensure assignments bridge academic concepts with workplace implementation\n"
        
        "You will analyze example assignments for format and structure reference, but your new assignments must:\n"
        "- Address real workplace challenges within the student's industry domain\n"
        "- Create tangible outputs that add value to their organization\n"
        "- Develop skills that enhance their professional effectiveness\n"
        "- Connect theoretical knowledge to practical application scenarios\n"
        
        "Maintain professional tone and clear structure while prioritizing practical utility and industry relevance."
    )

    user_prompt = (
        f"{examples_text}\n\n"
        f"### Target Assignment Context\n"
        f"Course Title: {course_title}\n"
        f"Academic Topic: {topic}\n"
        f"Student Industry Domain: {user_domain}\n"
    )

    if extra_instructions:
        user_prompt += f"Instructor's Specific Requirements: {extra_instructions}\n"

    user_prompt += (
        "\n### Assignment Generation Guidelines\n"
        "Create a NEW, industry-focused assignment that:\n\n"
        
        "**PRACTICAL VALUE REQUIREMENTS:**\n"
        "- Addresses a real challenge/opportunity in the {user_domain} industry\n"
        "- Produces deliverables the student can implement in their current workplace\n"
        "- Develops skills directly applicable to their job role and career advancement\n"
        "- Creates solutions that add measurable value to their organization\n\n"
        
        "**INDUSTRY INTEGRATION:**\n"
        "- Use industry-specific terminology, tools, and methodologies from {user_domain}\n"
        "- Reference current industry trends, standards, and best practices\n"
        "- Include scenarios that mirror actual workplace situations\n"
        "- Connect academic concepts to practical business outcomes\n\n"
        
        "**ASSIGNMENT STRUCTURE:**\n"
        "- Include 2-4 progressive questions/tasks that build practical competency\n"
        "- Use clear bullet points or numbering for organization\n"
        "- Provide specific, actionable deliverables\n"
        "- Include implementation guidance for workplace application\n\n"
        
        "**OUTPUT FORMAT:**\n"
        "- Title that reflects the practical focus and industry relevance\n"
        "- Brief context explaining the real-world scenario\n"
        "- Numbered tasks with clear deliverable expectations\n"
        "- Professional, concise language suitable for working professionals\n\n"
        
        "Generate an assignment that students will find immediately useful in their professional role within the {user_domain} industry."
    )

    print("Calling LLM for assignment generation...")
    llm_response = await invoke_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_type=ModelType.ANALYSIS
    )

    assignment_text = llm_response.get("answer", "")

    # --- Save generated assignment ---
    new_content = AssignmentContent(assignment_text=assignment_text)
    db.add(new_content)
    db.commit()
    db.refresh(new_content)

    return {
        "generated_assignment": assignment_text,
        "assignment_id": new_content.id
    }

def generate_rubric_from_assignment(db: Session, assignment_id: int):
    assignment = db.query(AssignmentContent).filter_by(id=assignment_id).first()
    if not assignment:
        return {"error": "Assignment not found"}

    response = requests.post(RUBRIC_API_URL, json={
        "text": assignment.assignment_text,
        "doc_type": "Situated Learning Assignment"
    })

    if response.status_code != 200:
        return {"error": "Rubric generation failed", "details": response.text}

    rubric = response.json().get("rubric")

    assignment.rubric = json.dumps(rubric)
    db.commit()
    db.refresh(assignment)
    return {"assignment_id": assignment.id, "rubric": json.dumps(rubric,indent=2)}



@app.post("/assignments/{assignment_id}/generate_rubric")
def generate_rubric_endpoint(assignment_id: int, db: Session = Depends(get_db)):
    return generate_rubric_from_assignment(db, assignment_id)



# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class EvaluationRequest(BaseModel):
    rubric: Dict
    assignment_description: str
    submission: str

class EvaluationReportResponse(BaseModel):
    rubric_name: str
    doc_type: str
    evaluation_summary: Dict
    category_breakdown: Dict
    detailed_results: List[Dict]

# Dependency to get evaluator instance
def get_evaluator():
    return AssignmentEvaluator()

@app.post("/api/evaluate_assignment")
async def evaluate_assignment(
    request: EvaluationRequest,
    evaluator: AssignmentEvaluator = Depends(get_evaluator)
):
    """
    Evaluate a student submission against a rubric using LLM
    
    - **rubric**: JSON rubric structure with 'rubrics' array containing categories and questions
    - **assignment_description**: Description of the assignment requirements
    - **submission**: Student's submission text content
    
    Returns comprehensive evaluation report with scores and justifications
    """
    try:
        # Validate rubric structure
        if not isinstance(request.rubric, dict):
            raise HTTPException(
                status_code=400, 
                detail="Rubric must be a JSON object"
            )
        
        if 'rubrics' not in request.rubric:
            raise HTTPException(
                status_code=400, 
                detail="Rubric must contain 'rubrics' key with evaluation criteria"
            )
        
        # Validate rubrics array
        rubrics = request.rubric.get('rubrics', [])
        if not isinstance(rubrics, list) or len(rubrics) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Rubric must contain non-empty 'rubrics' array"
            )
        
        # Validate each category has questions
        for i, category in enumerate(rubrics):
            if 'category' not in category:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Rubric category {i} missing 'category' field"
                )
            if 'questions' not in category or not isinstance(category.get('questions'), list) or len(category.get('questions', [])) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Category '{category.get('category')}' must have non-empty questions array"
                )

        # Validate other inputs
        if not request.assignment_description.strip():
            raise HTTPException(
                status_code=400, 
                detail="Assignment description cannot be empty"
            )
        
        if not request.submission.strip():
            raise HTTPException(
                status_code=400, 
                detail="Submission cannot be empty"
            )

        # Log evaluation start
        logger.info(f"Starting assignment evaluation with {len(rubrics)} categories")
        total_questions = sum(len(category.get('questions', [])) for category in rubrics)
        logger.info(f"Total questions to evaluate: {total_questions}")

        # Evaluate submission
        results = evaluator.evaluate_submission(
            request.assignment_description,
            request.submission,
            request.rubric
        )
        
        # Generate report
        report = evaluator.generate_report(
            results, 
            request.rubric, 
            "inline_submission"
        )
        
        # Remove file-related fields from response
        report.pop('submission_file', None)
        
        # Log successful completion
        summary = report['evaluation_summary']
        logger.info(
            f"Evaluation completed successfully. "
            f"Score: {summary['total_score']}/{summary['total_questions']} "
            f"({summary['percentage']:.1f}%)"
        )
        
        return report
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during evaluation: {str(e)}"
        )

# Health check endpoint
@app.post("/api/health")
async def health_check(evaluator: AssignmentEvaluator = Depends(get_evaluator)):
    """
    Health check for the evaluation service
    """
    return {
        "status": "healthy",
        "service": "assignment_evaluator",
        "model": evaluator.model,
        "base_url": evaluator.base_url
    }



# @app.post("/generate_answer_key")
# async def generate_answer_key(
#     assignment_text: str = Body(..., description="The assignment text to create answers for")
# ):
#     system_prompt = (
#         "You are an AI assistant that generates detailed answer keys for assignments. "
#         "The provided text is the assignment. "
#         "For each question in the assignment, provide a clear and precise answer. "
#         "Match the question numbering exactly. "
#         "If the question has multiple parts, label each part in the answer."
#     )

#     user_prompt = (
#         f"### Assignment\n"
#         f"{assignment_text}\n\n"
#         f"### Task\n"
#         f"Generate a complete and accurate answer key for the above assignment."
#     )

#     print("Calling LLM for answer key generation...")
#     llm_response = await invoke_llm(
#         system_prompt=system_prompt,
#         user_prompt=user_prompt,
#         model_type=ModelType.ANALYSIS
#     )

#     return {
#         "answer_key": llm_response.get("answer", "")
#     }



from database.connector import is_database_connected

@app.get("/db_status")
def db_status():
    if is_database_connected():
        return {"database_live": True}
    else:
        return {"database_live": False}


def main():

    # Start the FastAPI server
    config = uvicorn.Config("backend.src.main:app", host="0.0.0.0", port=8090, reload=True, log_level="info")
    server = uvicorn.Server(config)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    main()
