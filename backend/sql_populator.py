import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traceback
from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import declarative_base
from database.connector import engine, SessionLocal
from dotenv import load_dotenv
from database.models import File

# 🔹 Import your PDF extractor
from Question_Extractor.src.question_processor import process_single_pdf_with_verification

# --- CONFIG ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src/pdfs"))

llm_url = os.getenv("VLLM_URL_FOR_ANALYSIS")
llm_model = os.getenv("VLLM_MODEL_FOR_ANALYSIS")
verba_url = os.getenv("VERBA_URL")


# --- SQLAlchemy Models ---
Base = declarative_base()


# --- Create table if not exists ---
Base.metadata.create_all(bind=engine)

def populate_database():
    session = SessionLocal()
    try:
        files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".pdf")]
        if not files:
            print(f"[WARNING] No PDF files found in {BASE_DIR}")
            return

        for filename in files:
            file_path = os.path.join(BASE_DIR, filename)
            print(f"[INFO] Processing {file_path}...")

            try:
                # Extract text using your LLM PDF processor
                text = process_single_pdf_with_verification(
                    pdf_path=file_path,
                    model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
                    base_url="http://localhost:9091/v1/chat/completions/",
                )

                if not text.strip():
                    print(f"[WARNING] No content extracted from {filename}, skipping...")
                    continue

                # Insert into DB with correct columns
                db_file = File(
                    file_path=file_path,
                    description=os.path.splitext(filename)[0],  # filename without .pdf
                    content=text,
                    course_title=None  # leave empty unless you want to fill
                )
                session.add(db_file)
                session.commit()
                print(f"[SUCCESS] Inserted {filename} into DB")

            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
                traceback.print_exc()
                session.rollback()  # rollback after error so next file can continue

    finally:
        session.close()

if __name__ == "__main__":
    populate_database()
