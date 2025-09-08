from sqlalchemy.orm import Session
from database.connector import SessionLocal
from database.models import Minio


def save_metadata_to_db(
    file_name: str,
    file_type: str,
    file_size: int,
    minio_path: str,
    instructor_name: str,
    topic: str,
    course_code: str,       
    assignment_date: str,
):
    """
    Save assignment metadata into assignments_db.minio_assignments.
    """
    db: Session = SessionLocal()
    try:
        new_record = Minio(
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            minio_path=minio_path,
            instructor_name=instructor_name,
            topic=topic,
            course_code=course_code,  
            assignment_date=assignment_date,
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        print(
            f"✅ Saved to assignments_db.minio_assignments: "
            f"{new_record.id} | {file_name} -> {minio_path} | Course: {course_code}"
        )
    finally:
        db.close()

