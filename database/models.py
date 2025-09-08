from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, func, UniqueConstraint
from sqlalchemy.orm import relationship
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

class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    course_code = Column(String(50), unique=True, nullable=False)  # ✅ UNIQUE
    academic_year = Column(String(20), nullable=False)
    semester = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    assignments = relationship("Minio", back_populates="course")


class Minio(Base):
    __tablename__ = "minio_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_code = Column(String(50), ForeignKey("courses.course_code", ondelete="CASCADE"))  # ✅ FK to course_code

    file_name = Column(String(255), nullable=False)
    minio_path = Column(String(500), nullable=False)
    instructor_name = Column(String(255), nullable=False)
    topic = Column(String(255), nullable=False)
    assignment_date = Column(String(50), nullable=False)
    file_type = Column(String(50), nullable=True)
    file_size = Column(Integer, nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    course = relationship("Course", back_populates="assignments")

