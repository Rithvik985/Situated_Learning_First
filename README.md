

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Rithvik985/Situated_Learning_Final.git
```

### 2. Start MySQL using Docker

```bash
docker-compose up -d
```

Ensure MySQL is accessible on `localhost:3306` with credentials set in the `.env` or `src/configs` file (depending on your project).

Add all the required Course Metadata along with the path of the pdfs in the sql file in the init-sql folder. (Has two entries by default)

### 3. Place PDFs

Put all assignment PDF files inside:

```
backend/src/pdfs/
```

The PDF file paths should be preloaded or inserted into the MySQL `Assignment` table.

### 4. Start the LLMs (vLLM)

Ensure a vLLM server is running locally and listening at:

```
http://localhost:9091/v1
```

This LLM is used for assignment generation.

Also ensure to put your image model as well as the url for the image model in the .env file in the VLLM_MODEL_FOR_IMAGE and VLLM_URL_FOR_IMAGE

### 5. Start the FastAPI Backend (First Open env)

Enter Backend and open environment newestenv and download all from requirements.txt

```bash
cd backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8090
```

Backend will start at: [http://localhost:8090](http://localhost:8090)

### 6. Start the React Frontend

```bash
cd frontend
npm install
npm start
```

Frontend will start at: [http://localhost:3000](http://localhost:3000)

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Health check |
| `/llm_status` | GET | Check if LLM server is responding |
| `/db_status` | GET | Check if MySQL is connected |
| `/assignments/by_course/{course_id}` | GET | Fetch assignments by course |
| `/start_assignment_session` | POST | Start a generation session |
| `/generate_from_topic` | POST | Generate a new assignment based on topic |

---

## 📁 Database: MySQL

Make sure the `Assignment` table contains:

- `id`
- `course_id`
- `course_title`
- `instructor_name`
- `topic`
- `pdf_link` (relative path like `src/mysql-init/pdfs/myfile.pdf`)

---

## 🧠 LLM Server (vLLM)

Expected to run locally at `http://localhost:9091/v1`.  
Set it up separately via [vLLM docs](https://github.com/vllm-project/vllm).  
Recommended model: `Meta-Llama-3-70B-Instruct-AWQ` or similar.

---

## 🛠 Dev Tips

- Run backend and frontend separately
- Ensure LLM is active before generating assignments
- PDF extraction failures will be logged to console

---



