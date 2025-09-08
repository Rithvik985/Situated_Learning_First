from database.connector import engine
from database.models import *
from database.base import Base

def init_db():
    """
    Create all tables defined in Base metadata inside assignments_db.
    """
    Base.metadata.create_all(bind=engine)


def init_minio_table():
    """
    Create only the minio_assignments table inside assignments_db.
    """
    from database.models import Minio, Course
    Base.metadata.create_all(bind=engine, tables=[Course.__table__, Minio.__table__])

