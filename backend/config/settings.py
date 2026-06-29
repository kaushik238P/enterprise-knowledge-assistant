# backend/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # API Keys & General Configuration
    # ------------------------------------------------------------------
    google_api_key: str
    exa_api_key: str
    web_search_provider: str = "exa"

    # Optional future providers
    groq_api_key: str 
    mistral_api_key: str
    
    documents_path: str = "./docs"
    supported_document_types: tuple[str, ...] = (".pdf", ".txt", ".md")
    max_document_size_mb: int = 100
    ocr_enabled: bool = False
    table_extraction_enabled: bool = True

    # ------------------------------------------------------------------
    # Performance & Optimization Settings (NEW)
    # ------------------------------------------------------------------
    max_context_chars: int = 12000  # Configurable context size limit
    max_web_result_chars: int = 1200  # Exa webpage content truncation length
    enable_answer_evaluation: bool = True  # Skip evaluation to save 15-20s
    frontend_read_timeout: int = 180  # Streamlit client read timeout
    upload_timeout: int = 300  # Streamlit client document upload timeout

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    
    # ------------------------------------------------------------------
    # Chunking & Embeddings
    # ------------------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_words_per_element: int = 5
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------

    # Local development:
    #   http://localhost:6333
    #
    # Docker:
    #   http://qdrant:6333

    qdrant_url: str = "http://localhost:6333"

    # Optional for Qdrant Cloud
    qdrant_api_key: str | None = None

    qdrant_collection_name: str = "eka_documents"

    embedding_dimension: int = 384

    # ------------------------------------------------------------------
    # Retrieval & Reranker
    # ------------------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_min_score: float = 0.35
    grounding_threshold: float = 0.75
    query_coverage_threshold: float = 0.75
    fusion_top_k: int = 10
    tavily_search_depth: str = "advanced"
    sparse_model: str = "Qdrant/bm25"
    dense_top_k: int = 20
    hybrid_top_k: int = 50
    reranker_top_k: int = 30
    final_top_k: int = 5

    # Hybrid Retrieval Tuning Settings
    dense_rrf_weight: float = 0.65
    sparse_rrf_weight: float = 0.35
    metadata_bonus: float = 0.05
    query_expansion_enabled: bool = True
    duplicate_similarity_threshold: float = 0.85

    # Reranker & Retrieval Orchestration Tuning Settings
    candidate_pool_size: int = 50
    reranker_batch_size: int = 32
    cross_encoder_weight: float = 0.80
    hybrid_weight: float = 0.15
    metadata_weight: float = 0.05
    diversity_weight: float = 0.05
    reranker_normalization_strategy: str = "min-max"

    # Production Retriever Orchestration Settings
    adaptive_retrieval_enabled: bool = True
    enable_context_optimization: bool = True
    enable_retrieval_diagnostics: bool = True
    min_chunk_length: int = 30

    # Adaptive Complexity Parameter Groups
    adaptive_simple_hybrid_k: int = 20
    adaptive_simple_pool_size: int = 10
    adaptive_simple_final_k: int = 5

    adaptive_medium_hybrid_k: int = 40
    adaptive_medium_pool_size: int = 20
    adaptive_medium_final_k: int = 8

    adaptive_complex_hybrid_k: int = 50
    adaptive_complex_pool_size: int = 25
    adaptive_complex_final_k: int = 10

    # Production Context Builder / Generator Settings
    max_context_sources: int = 10
    max_context_tables: int = 5
    max_merged_chunk_size: int = 1200

    # Production Orchestrator Settings
    enable_evaluation: bool = True
    enable_response_metadata: bool = True
    max_query_length: int = 1000
    pipeline_timeout: float = 30.0
    enable_pipeline_diagnostics: bool = True

    # Production Evaluator Settings
    enable_grounding_check: bool = True
    enable_coverage_check: bool = True
    enable_citation_check: bool = True
    enable_numeric_check: bool = True
    enable_table_check: bool = True
    enable_hallucination_check: bool = True
    minimum_confidence: float = 0.70
    evaluation_timeout: float = 10.0
    hallucination_fail_level: str = "High"

    # Evaluator Controls
    enable_numeric_validation: bool = True
    enable_table_validation: bool = True
    enable_entity_validation: bool = True
    strict_grounding: bool = False
    strict_citation_validation: bool = False
    allow_missing_citations: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()