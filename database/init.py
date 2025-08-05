from database.connector import engine
from database.models import *
from database.base import Base

def init_db():
    Base.metadata.create_all(bind=engine)