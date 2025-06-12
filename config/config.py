from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    supabase_table_documents: str = Field("documents", env="SUPABASE_TABLE_DOCUMENTS")
    supabase_table_chat_history: str = Field("chat_history", env="SUPABASE_TABLE_CHAT_HISTORY")
    supabase_table_pdf_ingestion: str = Field("pdf_ingestion", env="SUPABASE_TABLE_PDF_INGESTION")

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    # RAG params
    top_k: int = Field(5, env="TOP_K")
    pdf_dir: str = Field("pdfs/", env="PDF_DIR")
    class Config:
        env_file = ".env"

settings = Settings()