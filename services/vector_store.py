from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from config.config import settings
from db.supabase_client import create_supabase_client

# 1) Build a Supabase client from your URL and Key
supabase = create_supabase_client()

# 2) Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
)

# 3) Now pass the client into SupabaseVectorStore
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name=settings.supabase_table,  # e.g. "documents"
)