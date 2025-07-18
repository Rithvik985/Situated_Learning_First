from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from backend.src.utils import *
from typing import Dict, List, Optional, Union, Any, Literal
from backend.src.data_types_class import *
from fastapi.encoders import jsonable_encoder
from backend.src.configs import *
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator



app = FastAPI(
    title="Operations Chatbot App",
    description="Backend Operations Chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CancellationToken:
    def __init__(self):
        self.is_cancelled = False
        self.ws_closed = False

    def cancel(self):
        self.is_cancelled = True

    def mark_closed(self):
        self.ws_closed = True

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running on port 8090!"}


async def process_context(context_payload: Dict, query_context: Optional[str]) -> Dict:
    retrieve_response = await call_spanda_retrieve(context_payload)
    retrieved_context = retrieve_response.get('context', '')
    documents = retrieve_response.get('documents', [])

    document_titles = [doc['title'] for doc in documents]
    titles_string = ', '.join(document_titles) if documents else "No source documents"

    filtered_context = await context_relevance_filter(context_payload["query"], retrieved_context)

    final_combined_context = f"{filtered_context.strip()} {query_context.strip()}" if query_context else filtered_context.strip()

    return {
        "filtered_context": final_combined_context,
        "titles_string": titles_string
    }


async def generate_professional_response_streaming(filtered_context: str, user_query: str) -> AsyncGenerator[str, None]:
    print("Generating streamed response...")

    system_prompt = (
        "You are an operations assistant at an educational institution. "
        "Your job is to provide helpful, clear, and professional replies using only the given context. "
        "Keep the tone human and natural — like a knowledgeable colleague would explain it. "
        "If the answer isn't available in the context, say so honestly without making assumptions."
    )

    user_prompt_combined = (
        f"Context:\n{filtered_context.strip()}\n\n"
        f"Question:\n{user_query.strip()}\n\n"
        f"Write a helpful, professional, and human-like reply using only the context above. "
        f"Do not start with phrases like 'According to the context' or 'Based on the information provided'."
    )

    async for chunk in stream_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt_combined,
        model_type=ModelType.ANALYSIS,
        cancellation_token=CancellationToken()
    ):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        print(text, end="", flush=True)  # clean debug print
        yield text + " "  # add space for readability


@app.post("/operations_chatbot/get_solution")
async def solution_fetch(request: QueryRequest):
    try:
        context_payload = QueryPayload(
            query=request.userquery,
            RAG=RagConfigForGeneration["rag_config"],
            labels=[],
            documentFilter=[],
            credentials=credentials_default["credentials"]
        )

        print("request.userquery", request.userquery)

        context_result = await process_context(
            jsonable_encoder(context_payload),
            None
        )

        filtered_context = context_result.get("filtered_context")
        if not filtered_context:
            async def fallback_stream() -> AsyncGenerator[str, None]:
                fallback_text = (
                    "I'm unable to solve your query at the moment. "
                    "Please contact support at help@bits.pilani or call +91 98765 6789 for assistance."
                )
                for word in fallback_text.split(" "):
                    yield word + " "
                    await asyncio.sleep(0.05)  # Simulate token streaming

            return StreamingResponse(fallback_stream(), media_type="text/plain")

        # Stream the final answer chunk by chunk
        return StreamingResponse(
            generate_professional_response_streaming(
                filtered_context=filtered_context,
                user_query=context_payload.query
            ),
            media_type="text/plain"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def main():

    # Start the FastAPI server
    config = uvicorn.Config("backend.src.main:app", host="0.0.0.0", port=8090, reload=True, log_level="info")
    server = uvicorn.Server(config)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    main()
