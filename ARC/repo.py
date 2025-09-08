# ARC/repo.py

import os
import mimetypes
from ARC.minio_upload import upload_file
from database.connector import is_database_connected
from ARC.table import save_metadata_to_db


def assignment_repository(
    file_path: str,
    course_code: str,       
    instructor_name: str,
    topic: str,
    assignment_date: str,     
) -> bool:
    """
    Handles the complete process of:
      1. Uploading a file to MinIO
      2. Saving its metadata into assignments_db.minio_assignments
    """

    # ✅ Check DB connection (assignments_db)
    if not is_database_connected():
        raise ConnectionError("❌ Database connection failed. Please check your settings.")

    # ✅ Derive filename from path
    original_file_name = os.path.basename(file_path)

    # ✅ Auto-detect file size
    file_size = os.path.getsize(file_path)

    # ✅ Auto-detect file type (e.g., "application/pdf", "image/png")
    file_type, _ = mimetypes.guess_type(file_path)
    file_type = file_type or "unknown"

    # ✅ Step 1: Upload the file to MinIO
    minio_file_path = upload_file(file_path)

    # ✅ Step 2: Save metadata to assignments_db.minio_assignments
    save_metadata_to_db(
        file_name=original_file_name,   
        file_type=file_type,
        file_size=file_size,
        minio_path=minio_file_path,     
        instructor_name=instructor_name,
        topic=topic,
        course_code=course_code,       
        assignment_date=assignment_date 
    )

    print(f"✅ Repository update complete for {original_file_name}")
    return True


if __name__ == "__main__":
    # Example usage
    try:
        file_path = r"C:\Users\Varun Gaming\Spanda\Situated_Learning_First\ARC\4498495_85300_assignment 1.pdf"

        result = assignment_repository(
            file_path=file_path,
            course_code="FIN169",               
            instructor_name="John Doe",
            topic="Entrepreneurship",
            assignment_date="2024-09-05",       
        )
        if result:
            print("✅ Assignment processed successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")
