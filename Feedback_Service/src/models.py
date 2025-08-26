from sqlalchemy import Column, Integer, Text, TIMESTAMP, String
from sqlalchemy.sql import func
from .database import Base

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    feedback_type = Column(Text, index=True)  # "assignment", "rubric", "evaluation"
    rating = Column(Integer, nullable=False)
    suggestion = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    generated_content = Column(Text, nullable=False)