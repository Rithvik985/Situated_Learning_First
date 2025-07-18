from setuptools import setup, find_packages

setup(
    name='spanda_wilp_chatbot',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'annotated-types==0.7.0',
        'anyio==4.9.0',
        'click==8.2.1',
        'colorama==0.4.6',
        'exceptiongroup==1.3.0',
        'fastapi==0.116.0',
        'h11==0.16.0',
        'idna==3.10',
        'pydantic==2.11.7',
        'pydantic_core==2.33.2',
        'sniffio==1.3.1',
        'starlette==0.46.2',
        'typing-inspection==0.4.1',
        'typing_extensions==4.14.1',
        'uvicorn==0.35.0',
        'httpx==0.27.2',
        'dotenv',
        'aiohttp',
        'PyMuPDF==1.24.10',
        'python-docx==1.1.2',  
        'pytesseract==0.3.13',
        'Pillow==10.4.0',
        'langchain-text-splitters',
        'langchain',
        'langchain-core==0.3.6',
        'langchain==0.3.1',
        'langchain-ollama==0.2.0'


    ],
    entry_points={
        'console_scripts': [
            "operations-start = backend.src.main:main" 
        ],
    },
)
