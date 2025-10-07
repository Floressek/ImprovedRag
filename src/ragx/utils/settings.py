from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


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

    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "ragx_documents_v2")
    embedding_dim: int = int(os.getenv("QDRANT_EMBEDDING_DIM", "768"))
    distance_metric: str = os.getenv("QDRANT_DISTANCE_METRIC", "cosine")
    timeout_s: int = int(os.getenv("QDRANT_TIMEOUT_S", "60"))
    recreate_collection: bool = str_to_bool(os.getenv("QDRANT_RECREATE_COLLECTION", "false"))


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
    device: str = os.getenv("LLM_DEVICE", "auto")
    load_in_4bit: bool = str_to_bool(os.getenv("LLM_LOAD_IN_4BIT", "true"))
    max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "300"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "80"))
    rerank_top_m: int = int(os.getenv("RERANK_TOP_M", "50"))
    context_top_n: int = int(os.getenv("CONTEXT_TOP_N", "6"))


@dataclass
class HNSWConfig:
    """HNSW index configuration."""
    m: int = int(os.getenv("HNSW_M", "32"))
    ef_construct: int = int(os.getenv("HNSW_EF_CONSTRUCT", "256"))
    on_disk: bool = str_to_bool(os.getenv("HNSW_ON_DISK", "true"))
    search_ef: int = int(os.getenv("HNSW_SEARCH_EF", "128"))


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "1"))
    reload: bool = str_to_bool(os.getenv("API_RELOAD", "false"))


# @dataclass
# class GPUConfig:
#     """GPU configuration."""
#     cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
# pytorch_cuda_alloc_conf: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")


@dataclass
class HuggingFaceConfig:
    """HuggingFace configuration."""
    hf_home: str = os.getenv("HF_HOME", "./models/huggingface")
    transformers_cache: str = os.getenv("TRANSFORMERS_CACHE", "./models/transformers")
    hf_hub_cache: str = os.getenv("HF_HUB_CACHE", "./models/hub")


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
    # gpu: GPUConfig
    huggingface: HuggingFaceConfig

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
            # gpu=GPUConfig(),
            huggingface=HuggingFaceConfig(),
        )

    def setup_huggingface_cache(self) -> None:
        """Setup HuggingFace cache directories as environment variables."""
        os.environ["HF_HOME"] = self.huggingface.hf_home
        os.environ["TRANSFORMERS_CACHE"] = self.huggingface.transformers_cache
        os.environ["HF_HUB_CACHE"] = self.huggingface.hf_hub_cache


settings = Settings.load()
settings.setup_huggingface_cache()
