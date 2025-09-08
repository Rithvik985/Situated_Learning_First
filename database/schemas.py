from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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

class MinioCreate(BaseModel):
    original_file_name: str
    file_type: str
    file_size: int
    minio_file_path: str
    instructor_name: str
    topic: str
    course_code: str   
    assignment_id: str


class MinioOut(MinioCreate):
    id: int   
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class CourseCreate(BaseModel):
    title: str
    course_code: str
    academic_year: str
    semester: int
    description: Optional[str] = None


class CourseOut(CourseCreate):
    id: int  
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True