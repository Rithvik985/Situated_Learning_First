from sqlalchemy import Column, Integer, String, Text, ForeignKey
from database.base import Base

class Assignment(Base):
    __tablename__ = 'assignments'

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(String(255), nullable=False)
    course_title = Column(String(255), nullable=False)
    instructor_name = Column(String(255), nullable=False)
    pdf_link = Column(String(512), nullable=False)
    topic = Column(String(100), nullable=False)


class AssignmentContent(Base):
    __tablename__ = 'assignment_content'

    id = Column(Integer, primary_key=True, index=True)
    assignment_text = Column(Text, nullable=True)
    rubric = Column(Text, nullable=True)

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    file_path = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    course_title = Column(Text, nullable=True)