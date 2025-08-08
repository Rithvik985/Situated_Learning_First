from sqlalchemy import Column, Integer, String, Text
from database.base import Base

class Assignment(Base):
    __tablename__ = 'assignments'
    id = Column(Integer, primary_key=True)
    course_id = Column(String(255), nullable=False)
    course_title = Column(String(255), nullable=False)
    instructor_name = Column(String(255), nullable=False)
    pdf_link = Column(String(512), nullable=False)
    topic = Column(String(100), nullable=False)