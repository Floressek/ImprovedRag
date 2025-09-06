from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, List, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SearchParams,
    VectorParams,
)

logger = logging.getLogger(__name__)

IdLike = Union[str, int]
Vector = Sequence[float]


class QuadrantStore:
    """Quadrant vector store implementation."""

    def __init__(
            self,
            url: str = "http://localhost:6333",
            api_key: Optional[str] = None,
            collection_name: str = "ragx_documents",
            embedding_dim: int = 768,
            distance_metric: str = "cosine",  # 'cosine' | 'euclidean' | 'dot'
            recreate_collection: bool = False,
            timeout_s: int = 60,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance_metric = self._to_distance(distance_metric)

        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=timeout_s)
        self._ensure_collection(recreate_collection)

    @staticmethod
    def _to_distance(metric: str) -> Distance:
        m = metric.lower().strip()
        if m in ("cos", "cosine"):
            return Distance.COSINE
        if m in ("l2", "euclid", "euclidean"):
            return Distance.EUCLID
        if m in ("dot", "ip", "inner"):
            return Distance.DOT
        raise ValueError(f"Unknown distance metric: {metric}")

    def _ensure_collection(self, recreate: bool) -> None:
        exists = any(
            c.name == self.collection_name
            for c in self.client.get_collections().collections
        )

        if exists and recreate:
            logger.info(f"Recreating collection '{self.collection_name}'")
            self.client.delete_collection(collection_name=self.collection_name)
            exists = False

        if exists:
            info = self.client.get_collection(collection_name=self.collection_name)
            size = info.config.params.vectors.size
            if size != self.embedding_dim:
                raise ValueError(
                    f"Collection '{self.collection_name}' exists with different embedding size "
                    f"({size} != {self.embedding_dim}). Use recreate_collection=True to recreate."
                )
            return

        logger.info("Creating collection '%s'", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=self.distance_metric,
            ),
            optimizers_config=None,
            shard_number=1,
            replication_factor=1,
            write_consistency_factor=1,
        )
        self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """Create useful payload indexes (id/title as KEYWORD, numeric as INTEGER)."""
        schemes = {
            "doc_id": PayloadSchemaType.KEYWORD,
            "doc_title": PayloadSchemaType.KEYWORD,
            "position": PayloadSchemaType.INTEGER,
            "total_chunks": PayloadSchemaType.INTEGER,
        }
        for field, schema in schemes.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
                logger.debug("Created payload index on %s (%s)", field, schema)
            except Exception as e:
                logger.debug("Failed to create payload index for field '%s': %s", field, e)
