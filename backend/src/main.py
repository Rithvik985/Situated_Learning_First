import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import FastAPI, HTTPException, Request, Body
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
from database.init import init_db
from sqlalchemy.orm import Session
from fastapi import Depends
from sqlalchemy.orm import Session

from database.models import Assignment
from database.deps import get_db
from database.connector import *
# Add this import to fix the ModelType issue
from InferenceEngine.inference_engines import ModelType
# Place this at top of main.py
from uuid import uuid4
session_store = {}


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


@app.on_event("startup")
async def startup_event():
    init_db()

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


@app.get("/assignments/by_course/{course_id}")
def get_assignments_by_course(course_id: str):
    db: Session = SessionLocal()
    try:
        assignments = db.query(Assignment).filter(Assignment.course_id == course_id).all()
        return [
            {
                "id": a.id,
                "course_title": a.course_title,
                "instructor_name": a.instructor_name,
                "content": a.content,
                "topic": a.topic
            } for a in assignments
        ]
    finally:
        db.close()


@app.post("/start_assignment_session")
def start_assignment_session(course_id: str = Body(...)):
    db: Session = SessionLocal()
    try:
        assignments = db.query(Assignment).filter(Assignment.course_id == course_id).all()

        if not assignments:
            raise HTTPException(status_code=404, detail="No assignments found for this course.")

        examples = [
            {
                "course_title": a.course_title,
                "instructor": a.instructor_name,
                "topic": a.topic,
                "content": a.content.strip()
            }
            for a in assignments
        ]

        session_id = str(uuid4())
        session_store[session_id] = {
            "course_id": course_id,
            "examples": examples
        }

        return {
            "message": f"Session started for course_id={course_id}",
            "session_id": session_id
        }
    finally:
        db.close()

@app.post("/generate_from_topic")
async def generate_from_topic(
    session_id: str = Body(...),
    topic: str = Body(...)
):
    session_data = session_store.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    examples = session_data["examples"]
    course_id = session_data["course_id"]

    examples_text = "\n\n".join([
        f"### Example Assignment\n"
        f"Course Title: {ex['course_title']}\n"
        f"Instructor: {ex['instructor']}\n"
        f"Topic: {ex['topic']}\n"
        f"Content:\n{ex['content']}"
        for ex in examples
    ])

    system_prompt = (
        "You are an AI assistant that helps professors create new assignments. "
        "You will be shown several example assignments. Each example includes course information and a topic. "
        "Your task is to generate a new, original assignment that is similar in tone, structure, and difficulty, "
        "but does not repeat or paraphrase any of the example content. "
        "Use a professional tone, and avoid generic or vague tasks."
    )

    user_prompt = (
        f"{examples_text}\n\n"
        f"### Target Assignment\n"
        f"Course ID: {course_id}\n"
        f"Topic: {topic}\n"
        f"Now generate a new, original assignment for this course and topic. "
        f"Include 2–4 questions. Use bullet points or numbering. Be concise and clear."
    )

    print("Calling LLM for assignment generation...")
    llm_response = await invoke_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_type=ModelType.ANALYSIS
    )

    return {"generated_assignment": llm_response.get("answer", "")}

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
