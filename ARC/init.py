def init_minio_table():
    """
    Create the minio_assignments table inside assignments_db only.
    """
    from database.models import Minio  # import here to avoid circular import
    Base.metadata.create_all(bind=engine, tables=[Minio.__table__])