# schemas.py
from pydantic import BaseModel

class FeedbackCreate(BaseModel):
    feedback_type: str
    rating: int
    suggestion: str | None = None
    generated_content: str   # assignment/rubric/evaluation text

class FeedbackResponse(BaseModel):
    id: int
    feedback_type: str
    rating: int
    suggestion: str | None
    generated_content: str

    class Config:
        orm_mode = True
