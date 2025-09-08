import os
import sys
from minio import Minio
from datetime import datetime

# -------------------------
# MinIO Configuration (Docker default)
# -------------------------
MINIO_ENDPOINT = "localhost:9000"   # If Docker is mapped to 9000
MINIO_ACCESS_KEY = "minioadmin"          # Replace with your credentials
MINIO_SECRET_KEY = "minioadmin123"
BUCKET_NAME = "assignments"

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False   # Docker MinIO usually runs without TLS
)

# Create bucket if not exists
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)
    print(f"✅ Created bucket: {BUCKET_NAME}")


# -------------------------
# Upload File Function
# -------------------------
def upload_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    file_name = os.path.basename(file_path)

    # Generate object path (organized by date)
    today = datetime.today().strftime("%Y/%m/%d")
    minio_object_path = f"{today}/{file_name}"

    # Upload file
    minio_client.fput_object(BUCKET_NAME, minio_object_path, file_path)
    print(f"✅ Uploaded to MinIO → {minio_object_path}")

    return minio_object_path
