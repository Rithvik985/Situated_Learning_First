from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session

import src.models as models
import src.database as database
import src.schemas as schemas
from src.models import *
from src.schemas import *

app = FastAPI(title="Feedback Service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # must include OPTIONS
    allow_headers=["*"],
)
# Create tables
models.Base.metadata.create_all(bind=database.engine)

@app.post("/feedback", response_model=schemas.FeedbackResponse)
def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(database.get_db)):
    new_feedback = models.Feedback(
        feedback_type=feedback.feedback_type,
        rating=feedback.rating,
        suggestion=feedback.suggestion,
        generated_content=feedback.generated_content
    )
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    return new_feedback
    return {"message": "Feedback submitted", "id": new_feedback.id}


@app.get("/feedback", response_model=list[schemas.FeedbackResponse])
def list_feedback(
    feedback_type: str | None = Query(None), 
    db: Session = Depends(database.get_db)
):
    query = db.query(models.Feedback)
    if feedback_type:
        query = query.filter(models.Feedback.feedback_type == feedback_type)
    return query.all()