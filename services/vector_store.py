from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from config.config import settings

# 1) Build a Supabase client from your URL and Key
supabase_client = create_client(
    settings.supabase_url,   # e.g. "https://abcd1234.supabase.co"
    settings.supabase_key    # service_role or anon key
)

# 2) Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
)

# 3) Now pass the client into SupabaseVectorStore
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name=settings.supabase_table,  # e.g. "documents"
)