from setuptools import setup, find_packages

setup(
    name="situated_learning",
    version="1.0.0",
    description="FastAPI backend for Situated Learning App with MySQL, Ollama, and VLLM integration",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="."),  # Includes src/, database/, etc.
    include_package_data=True,
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "sqlalchemy>=2.0.0",
        "pymysql>=1.0.3",
        "python-dotenv>=1.0.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "pillow>=10.0.0",
        "python-multipart",  # for file uploads
        "PyMuPDF",           # fitz
        "langchain",         # if used for chunking/splitting
        "sentence-transformers",  # for embedding
        "openpyxl",          # if you load Excel files
        "mammoth",           # if you ingest .doc files
        "docx2txt",
        "extract-msg",       # for .msg files
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "run-backend=main:main",  # allows running with: `run-backend`
        ]
    }
)
