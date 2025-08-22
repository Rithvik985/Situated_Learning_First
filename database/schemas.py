from pydantic import BaseModel
from typing import Optional

# --- Assignment ---
class AssignmentCreate(BaseModel):
    course_id: str
    course_title: str
    instructor_name: str
    pdf_link: str
    topic: str

class AssignmentOut(AssignmentCreate):
    id: int
    class Config:
        orm_mode = True

# --- Assignment Content ---
class AssignmentContentCreate(BaseModel):
    assignment_text: Optional[str]
    rubric: Optional[str]

class AssignmentContentOut(AssignmentContentCreate):
    id: int
    class Config:
        orm_mode = True

# --- File ---
class FileCreate(BaseModel):
    file_path: str
    description: Optional[str]
    content: Optional[str]
    course_title: Optional[str]

class FileOut(FileCreate):
    id: int
    class Config:
        orm_mode = True
