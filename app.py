import os
from typing import Any, Tuple
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware   
from sqlalchemy import text
from services.schemas import QueryRequest, QueryResponse, ChatHistoryItem
from typing import Any, List, Dict
import logging
from fastapi import Query
from supabase import Client, create_client
from services.config import settings
from fastapi.responses import JSONResponse
from postgrest import APIError
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import openai

logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG FastAPI Supabase API",
    version="1.0.0",
    description="RAG service using Supabase vector store and OpenAI API",
)

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Origin",
        "X-Requested-With",
        "Content-Type",
        "Accept",
        "Authorization",
        "X-HTTP-Method-Override",
    ],
)

def create_supabase_client():
    supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
    return supabase

supabase = create_supabase_client()

router_v1 = APIRouter(prefix="/v1")

@router_v1.get("/test-db")
def test_db():
    try:
        result = supabase\
            .table("documents")\
            .select("id")\
            .limit(1)\
            .execute()
        return {"status": "ok", "result": result.data}
    
    except APIError as e:
        # Supabase-py wraps server and parsing errors in APIError
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=f"Supabase API error: {e}"
        )

    except Exception as e:
        # Any other unexpected error
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router_v1.get(
    "/documents",
    summary="List documents with pagination",
    description="Fetches paginated rows from the Supabase 'documents' table.",
    response_model=List[Dict[str, Any]],
)
def get_all_documents(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)) -> Any:
    try:
        logger.info(f"Fetching documents: skip={skip}, limit={limit}")
        result = supabase\
            .table("documents")\
            .select("id, content, embedding, metadata")\
            .limit(limit)\
            .offset(skip)\
            .execute()
        logger.info(f"Received {len(result.data)} documents")
        return result.data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@router_v1.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Retrieval-Augmented Generation over ingested documents."
)
def query_qa(req: QueryRequest) -> QueryResponse:
    answer, sources = answer_question(req.question)
    return QueryResponse(answer=answer, source_docs=sources)

openai.api_key = settings.openai_api_key
embedding_model = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key
)

def to_pgvector_literal(vec: list[float]) -> str:
    return f"[{','.join(f'{x:.6f}' for x in vec)}]"

def answer_question(question: str) -> Tuple[str, List[Dict[str, any]]]:
    # 1) Embed the question
    q_vector = embedding_model.embed_query(question)
    # to_pgvector_literal should turn [0.1, 0.2, …] into a Postgres‐literal like
    # "'[0.1,0.2,…]'" so you can inline it into your select.
    q_vector_literal = to_pgvector_literal(q_vector)

    # 2) Fetch top‐5 by cosine similarity using the pgvector operator <=>
    select_str = (
         f"id, content, metadata, "
         f"similarity:1-(embedding<=>{q_vector_literal})"
    )

    resp = (
        supabase
        .table("documents")
        .select(select_str)
        .order("similarity", desc=True)   # order by the aliased field
        .limit(5)
        .execute()
    )

    rows = resp.data  # List[dict]

    # 3) Build context & sources list
    context_blocks = [r["content"] for r in rows]
    source_docs = [
        {"id": str(r["id"]), "similarity": float(r["similarity"]), "metadata": r["metadata"]}
        for r in rows
    ]

    context = "\n\n---\n\n".join(context_blocks)
    prompt = (
        "Use the following context to answer the question:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # 4) Call OpenAI synchronously
    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = completion.choices[0].message.content.strip()

    return answer, source_docs


@router_v1.get(
    "/history/{conversation_id}",
    response_model=List[ChatHistoryItem],
    tags=["History"],
    summary="Get chat history for a conversation",
    description="Returns an array of { question, answer } for the given conversation_id"
)
async def read_history(conversation_id: str):
    try:
        logger.info(f"Fetching history")
        result = supabase\
            .table("chat_history")\
            .select("id, conversation_id, question, answer")\
            .eq("conversation_id", conversation_id)\
            .execute()

        data = []
        for row in result.data:
            row["id"] = str(row["id"])
            data.append(row)

        logger.info(f"Received {len(result.data)} chat histories")
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

app.include_router(router_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))