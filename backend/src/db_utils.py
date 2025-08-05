# import os
# import sqlalchemy
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from src.configs import SQLALCHEMY_DATABASE_URL
# from src.data_types_class import Base

# MYSQL_USER = os.getenv('MYSQL_USER', 'assignments_user')
# MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'assignments_pass')
# MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
# MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
# MYSQL_DB = os.getenv('MYSQL_DATABASE', 'assignments_db')

# SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"


# engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True, future=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# def init_db():
#     Base.metadata.create_all(bind=engine)


