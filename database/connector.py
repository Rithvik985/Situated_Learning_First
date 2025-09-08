# backend/database/connector.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from database.base import Base
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER", "assignments_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "assignments_pass")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DATABASE", "assignments_db")

SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def is_database_connected() -> bool:
    try:
        db = SessionLocal()
        db.execute(text('SELECT 1'))  # simple ping query
        return True
    except OperationalError:
        return False
    finally:
        db.close()
        
