
import sys
import os
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import models, deps, schemas
from database.connector import engine


# Create all tables
models.Base.metadata.create_all(bind=engine)

# FastAPI instance
app = FastAPI(title="Assignment Backend API")

# Enable CORS (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- ASSIGNMENTS --------------------
@app.post("/assignments/", response_model=schemas.AssignmentOut)
def create_assignment(assignment: schemas.AssignmentCreate, db: Session = Depends(deps.get_db)):
    db_assignment = models.Assignment(**assignment.dict())
    db.add(db_assignment)
    db.commit()
    db.refresh(db_assignment)
    return db_assignment

@app.get("/assignments/", response_model=list[schemas.AssignmentOut])
def get_assignments(db: Session = Depends(deps.get_db)):
    return db.query(models.Assignment).all()

# -------------------- ASSIGNMENT CONTENT --------------------
@app.post("/assignment_content/", response_model=schemas.AssignmentContentOut)
def create_assignment_content(content: schemas.AssignmentContentCreate, db: Session = Depends(deps.get_db)):
    db_content = models.AssignmentContent(**content.dict())
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content

@app.get("/assignment_content/", response_model=list[schemas.AssignmentContentOut])
def get_assignment_contents(db: Session = Depends(deps.get_db)):
    return db.query(models.AssignmentContent).all()

# -------------------- FILES --------------------
@app.post("/files/", response_model=schemas.FileOut)
def create_file(file: schemas.FileCreate, db: Session = Depends(deps.get_db)):
    db_file = models.File(**file.dict())
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

@app.get("/files/", response_model=list[schemas.FileOut])
def get_files(db: Session = Depends(deps.get_db)):
    return db.query(models.File).all()


from typing import Optional

@app.get("/assignments/search", response_model=list[schemas.AssignmentOut])
def search_assignments(
    course_id: Optional[str] = None,
    instructor_name: Optional[str] = None,
    topic: Optional[str] = None,
    db: Session = Depends(deps.get_db)
):
    query = db.query(models.Assignment)
    if course_id:
        query = query.filter(models.Assignment.course_id == course_id)
    if instructor_name:
        query = query.filter(models.Assignment.instructor_name.ilike(f"%{instructor_name}%"))
    if topic:
        query = query.filter(models.Assignment.topic.ilike(f"%{topic}%"))
    return query.all()


@app.get("/assignment_content/search", response_model=list[schemas.AssignmentContentOut])
def search_assignment_content(
    keyword: Optional[str] = None,
    has_rubric: Optional[bool] = None,
    db: Session = Depends(deps.get_db)
):
    query = db.query(models.AssignmentContent)
    if keyword:
        query = query.filter(models.AssignmentContent.assignment_text.ilike(f"%{keyword}%"))
    if has_rubric is not None:
        if has_rubric:
            query = query.filter(models.AssignmentContent.rubric.isnot(None))
        else:
            query = query.filter(models.AssignmentContent.rubric.is_(None))
    return query.all()

@app.get("/files/search", response_model=list[schemas.FileOut])
def search_files(
    course_title: Optional[str] = None,
    description: Optional[str] = None,
    db: Session = Depends(deps.get_db)
):
    query = db.query(models.File)
    if course_title:
        query = query.filter(models.File.course_title.ilike(f"%{course_title}%"))
    if description:
        query = query.filter(models.File.description.ilike(f"%{description}%"))
    return query.all()


# -------------------- MAIN FUNCTION --------------------
if __name__ == "__main__":
    import uvicorn
    # Run FastAPI with Uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=6023, reload=True)
