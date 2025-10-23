from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
root_dir = Path(__file__).parent.parent.parent.parent
env_path = root_dir / ".env"

load_dotenv(dotenv_path=env_path, override=False)


def str_to_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "t", "y")


@dataclass
class AppConfig:
    """General application configuration."""
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    data_dir: str = os.getenv("DATA_DIR", "./data")
    raw_dir: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    processed_dir: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")
    indices_dir: str = os.getenv("INDICES_DIR", "./data/indices")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models/cache")


@dataclass
class QdrantConfig:
    """Qdrant vector store configuration."""
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None

    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "ragx_documents_v3")
    embedding_dim: int = int(os.getenv("QDRANT_EMBEDDING_DIM", "768"))
    distance_metric: str = os.getenv("QDRANT_DISTANCE_METRIC", "cosine")
    timeout_s: int = int(os.getenv("QDRANT_TIMEOUT_S", "60"))
    recreate_collection: bool = str_to_bool(os.getenv("QDRANT_RECREATE_COLLECTION", "false"))
    max_retries: int = int(os.getenv("QDRANT_MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("QDRANT_RETRY_DELAY", "2.0"))


@dataclass
class EmbedderConfig:
    """Embedding model configuration."""
    model_id: str = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")
    device: str = os.getenv("EMBEDDING_DEVICE", "auto")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    max_seq_length: int = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))
    normalize_embeddings: bool = str_to_bool(os.getenv("EMBEDDING_NORMALIZE", "true"))
    show_progress: bool = str_to_bool(os.getenv("EMBEDDING_SHOW_PROGRESS", "true"))
    use_prefixes: bool = str_to_bool(os.getenv("EMBEDDING_USE_PREFIXES", "true"))

    query_prefix: str = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
    passage_prefix: str = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")

@dataclass
class ChunkerConfig:
    """Text chunker configuration."""
    strategy: str = os.getenv("CHUNKER_STRATEGY", "semantic")
    chunk_size: int = int(os.getenv("CHUNKER_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNKER_CHUNK_OVERLAP", "96"))
    min_chunk_size: int = int(os.getenv("CHUNKER_MIN_CHUNK_SIZE", "150"))
    max_chunk_size: int = int(os.getenv("CHUNKER_MAX_CHUNK_SIZE", "512"))
    respect_sections: bool = str_to_bool(os.getenv("CHUNKER_RESPECT_SECTIONS", "true"))
    breakpoint_percentile_threshold: int = int(os.getenv("CHUNKER_BREAKPOINT_PERCENTILE_THRESHOLD", "80"))
    buffer_size: int = int(os.getenv("CHUNKER_BUFFER_SIZE", "5"))
    add_passage_prefix: bool = str_to_bool(os.getenv("CHUNKER_ADD_PASSAGE_PREFIX", "false"))
    context_tail_tokens: int = int(os.getenv("CHUNKER_CONTEXT_TAIL_TOKENS", "0"))

    chunking_model: Optional[str] = os.getenv("CHUNKER_MODEL", None)



@dataclass
class RerankerConfig:
    """Reranker model configuration."""
    model_id: str = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    device: str = os.getenv("RERANKER_DEVICE", "auto")
    batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
    show_progress: bool = str_to_bool(os.getenv("RERANKER_SHOW_PROGRESS", "false"))


@dataclass
class LLMConfig:
    """LLM model configuration."""
    model_id: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    model_name: str = os.getenv("LLM_MODEL_NAME_OLLAMA", "Qwen/Qwen2.5-7B-Instruct")
    device: str = os.getenv("LLM_DEVICE", "cuda")
    load_in_4bit: bool = str_to_bool(os.getenv("LLM_LOAD_IN_4BIT", "true"))
    max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "2000"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))

    provider: str = os.getenv("LLM_PROVIDER", "huggingface")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_models_path: str = os.getenv("OLLAMA_MODELS_PATH", r"E:\Models\Ollama\.ollama\models")

    tensor_parallel_size: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    trust_remote_code: bool = str_to_bool(os.getenv("TRUST_REMOTE_CODE", "true"))
    quantization: Optional[str] = os.getenv("QUANTIZATION", "awq")

    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", "8192"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.1"))


@dataclass
class RetrievalConfig:
    """Retrieval pipelines configuration."""
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "100"))
    rerank_top_m: int = int(os.getenv("RERANK_TOP_M", "80"))
    context_top_n: int = int(os.getenv("CONTEXT_TOP_N", "8"))

@dataclass()
class RewriteConfig:
    """Rewrite configuration."""
    max_tokens: int = int(os.getenv("REWRITE_MAX_TOKENS", "1024"))
    temperature: float = float(os.getenv("REWRITE_TEMPERATURE", "0.2"))
    enabled: bool = str_to_bool(os.getenv("REWRITE_ENABLED", "true"))
    verify_before_retrieval: bool = str_to_bool(os.getenv("REWRITE_VERIFY_BEFORE_RETRIEVAL", "true"))


@dataclass
class HNSWConfig:
    """HNSW index configuration."""
    m: int = int(os.getenv("HNSW_M", "32"))
    ef_construct: int = int(os.getenv("HNSW_EF_CONSTRUCT", "256"))
    on_disk: bool = str_to_bool(os.getenv("HNSW_ON_DISK", "true"))
    search_ef: int = int(os.getenv("HNSW_SEARCH_EF", "128"))


@dataclass
class HuggingFaceConfig:
    """HuggingFace configuration."""
    hf_home: str = os.getenv("HF_HOME", "./models/huggingface")
    transformers_cache: str = os.getenv("TRANSFORMERS_CACHE", "./models/transformers")
    hf_hub_cache: str = os.getenv("HF_HUB_CACHE", "./models/hub")

@dataclass
class ChatConfig:
    """Chat configuration."""
    max_history: int = int(os.getenv("CHAT_MAX_HISTORY", "10"))
    system_prompt: str = os.getenv("CHAT_SYSTEM_PROMPT", "You are a helpful assistant.")
    context_window: int = int(os.getenv("CHAT_CONTEXT_WINDOW", "4096"))
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("CHAT_TOP_P", "0.9"))

@dataclass
class APIConfig:
    """API server configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "1"))
    reload: bool = str_to_bool(os.getenv("API_RELOAD", "true"))


@dataclass
class Settings:
    """Main settings object containing all configuration sections."""
    app: AppConfig
    qdrant: QdrantConfig
    embedder: EmbedderConfig
    chunker: ChunkerConfig
    reranker: RerankerConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    hnsw: HNSWConfig
    api: APIConfig
    huggingface: HuggingFaceConfig
    chat: ChatConfig

    @classmethod
    def load(cls) -> Settings:
        """Load settings from environment variables."""
        return cls(
            app=AppConfig(),
            qdrant=QdrantConfig(),
            embedder=EmbedderConfig(),
            chunker=ChunkerConfig(),
            reranker=RerankerConfig(),
            llm=LLMConfig(),
            retrieval=RetrievalConfig(),
            hnsw=HNSWConfig(),
            api=APIConfig(),
            huggingface=HuggingFaceConfig(),
            chat=ChatConfig(),
            rewrite=RewriteConfig()
        )

    def setup_huggingface_cache(self) -> None:
        """Setup HuggingFace cache directories as environment variables."""
        os.environ["HF_HOME"] = self.huggingface.hf_home
        os.environ["TRANSFORMERS_CACHE"] = self.huggingface.transformers_cache
        os.environ["HF_HUB_CACHE"] = self.huggingface.hf_hub_cache


settings = Settings.load()
settings.setup_huggingface_cache()


if __name__ == "__main__":
    print("=== Settings Debug ===")
    print(f"Root dir: {root_dir}")
    print(f".env path: {env_path}")
    print(f"Loaded Quadrant collection: {settings.qdrant.collection_name}")
    print(f".env exists: {env_path.exists()}")
    print(f"\nQdrant URL: {settings.qdrant.url}")
    print(f"Embedding model: {settings.embedder.model_id}")
    print(f"Chunk strategy: {settings.chunker.strategy}")
