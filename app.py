import os
from typing import Any, Tuple, Generator
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from services.schemas import QueryRequest, QueryResponse, ChatHistoryItem, UploadResponse
from typing import Any, List, Dict
import logging
from fastapi import Query
from config.config import settings
from fastapi.responses import JSONResponse, StreamingResponse
from postgrest import APIError
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store import vector_store
from db.supabase_client import create_supabase_client

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

@router_v1.get("/documents",
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

@router_v1.post("/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Retrieval-Augmented Generation over ingested documents."
)
def query_qa(req: QueryRequest) -> QueryResponse:
    answer, sources = answer_question(req.question, match_threshold=0.75, match_count=5)
    return QueryResponse(answer=answer, source_docs=sources)

@router_v1.get("/history/{conversation_id}",
    response_model=List[ChatHistoryItem],
    tags=["History"],
    summary="Get chat history for a conversation",
    description="Returns an array of { question, answer } for the given conversation_id"
)
def read_history(conversation_id: str):
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

@router_v1.post("/query-stream",
    response_model=None,
    tags=["RAG"],
    summary="Streamed Q&A with history"
)
def query_stream(req: QueryRequest):
    # 0) ensure we have a UUID to track this conversation
    if req.conversation_id:
        try:
            uuid.UUID(req.conversation_id)
            conversation_id = req.conversation_id
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation_id format (must be UUID)")
    else:
        conversation_id = str(uuid.uuid4())

    # 1) load history
    history = get_history(conversation_id)

    # 2) stream tokens, then append history at the end
    def event_generator():
        for token in stream_answer_sync(req.question, conversation_id, history):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain; charset=utf-8",
        headers={"x-conversation-id": conversation_id}
    )

@router_v1.post("/upload",
    response_model=UploadResponse,
    summary="Upload a PDF document",
    description="Ingests a PDF, splits into chunks, and stores embeddings in Supabase"
)
def upload_pdf(file: UploadFile = File(...)):
    # validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # read & save locally
    contents = file.file.read()   # sync read
    os.makedirs(settings.pdf_dir, exist_ok=True)
    path = os.path.join(settings.pdf_dir, file.filename)
    with open(path, "wb") as f:
        f.write(contents)

    # ingest
    count = ingest_pdf_sync(path)
    logger.info(f"Finished ingestion, inserted {count} chunks.")

    return UploadResponse(
        message="PDF ingested successfully",
        inserted_count=count
    )

#/*------- service methods -------*/
openai.api_key = settings.openai_api_key
embedding_model = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key
)

def to_pgvector_literal(vec: list[float]) -> str:
    return f"[{','.join(f'{x:.6f}' for x in vec)}]"

def answer_question(
    question: str,
    match_threshold: float = 0.8,
    match_count: int    = 5,
) -> Tuple[str, List[Dict[str, any]]]:
    # 1) Embed the question
    q_vector = embedding_model.embed_query(question)
    
    # 2) Call the RPC
    try:
        rpc_payload = {
            "query_embedding": q_vector,
            "match_threshold": match_threshold,
            "match_count": match_count,
        }
        resp = supabase.rpc("match_documents", rpc_payload).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB function error: {e}")

    rows = resp.data

    if not rows:
        return "I couldn't find any relevant documents.", []

    # 3) Build context and source list
    context = "\n\n---\n\n".join(r["content"] for r in rows)
    source_docs = [
        {
            "id":       str(r["id"]),
            "similarity": float(r["similarity"]),
            "metadata": r["metadata"],
        }
        for r in rows
    ]

    prompt = (
        "Use the following context to answer the question:\n\n"
        f"{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    # 4) Call OpenAI synchronously
    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = completion.choices[0].message.content.strip()

    return answer, source_docs

def get_history(conversation_id: str) -> List[Dict[str, str]]:
    """Fetch prior turns for this conversation from Supabase."""
    resp = (
        supabase
        .table("chat_history")
        .select("question, answer")
        .eq("conversation_id", conversation_id)
        .order("id", desc=False)
        .execute()
    )
    if hasattr(resp, "error") and resp.error:
        raise HTTPException(status_code=500, detail=resp.error.message)
    return [
        {"question": r["question"], "answer": r["answer"]}
        for r in resp.data
    ]

def stream_answer_sync(
    question: str,
    conversation_id: str,
    history: List[Dict[str,str]],
    match_threshold: float = 0.75,
    match_count: int = 5,
) -> Generator[str, None, None]:
    """Yield OpenAI tokens one-by-one, using Supabase RPC for semantic search."""
    # 1) embed & fetch top docs via our pg function
    q_vec = embedding_model.embed_query(question)
    try:
        rpc_payload = {
            "query_embedding": q_vec,
            "match_threshold": match_threshold,
            "match_count": match_count,
        }
        docs_resp = supabase.rpc("match_documents", rpc_payload).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB function error: {e}")

    docs = docs_resp.data or []
    # build context
    context = "\n\n---\n\n".join(d["content"] for d in docs)

    # build history block
    if history:
        hist_block = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in history
        )
    else:
        hist_block = "(no prior context)\n"

    prompt = (
        "You are a helpful assistant.\n\n"
        f"Conversation so far:\n{hist_block}\n\n"
        f"Context from documents:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # 2) stream from OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    stream = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer += delta
            yield delta

    # after streaming is done, append to history
    try:
        supabase.table("chat_history").insert({
            "conversation_id": conversation_id,
            "question": question,
            "answer": full_answer
        }).execute()
    except Exception as e:
        # we can’t stream more at this point, so just log or ignore
        logger.error(f"Failed to append chat history: {e}")

def ingest_pdf_sync(file_path: str) -> int:
    # 1) load & chunk
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks.")

    # 2) push embeddings to Supabase vector store
    vector_store.add_documents(chunks)
    logger.info("Added embeddings to Supabase vector store.")

    # 3) record ingestion metadata in Postgres via Supabase
    filename = os.path.basename(file_path)
    metadata = {"chunks": len(chunks), "path": file_path, "ingested_at": datetime.now(timezone.utc).isoformat()}
    resp = supabase\
        .table("pdf_ingestion")\
        .insert({
            "filename": filename,
            "metadata": metadata
        })\
        .execute()

    if hasattr(resp, "error") and resp.error:
        # if the insert failed, surface an HTTP error
        raise HTTPException(status_code=500, detail=f"Failed to record ingestion: {resp.error.message}")

    logger.info("Inserted ingestion record into Supabase.")
    return len(chunks)

app.include_router(router_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))