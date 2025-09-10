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
        """
        Ensure the collection exists with the correct configuration.
        If recreate is True, delete and recreate the collection if it exists.
        """
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

    def add(
            self,
            vectors: Sequence[Vector],
            payloads: Sequence[dict[str, Any]],
            ids: Optional[Sequence[IdLike]] = None,
            batch_size: int = 1024,
            wait: bool = True,
    ) -> None:
        """
        Soft shell for upsert with some checks and auto ID generation.
        Args:
            vectors: list of vectors to add
            payloads: list of payload dicts (metadata)
            ids: optional list of IDs (if None, UUIDs will be generated)
            batch_size: number of points to upsert in each batch
            wait: whether to wait for the operation to complete
        """
        if len(vectors) != len(payloads):
            raise ValueError("Vectors and payloads must have the same length")

        n = len(vectors)
        if ids is None:
            ids = [str(uuid4()) for _ in range(n)]
        elif len(ids) != n:
            raise ValueError("IDs must have the same length as vectors and payloads")

        # Vector length check
        if n and len(vectors[0]) != self.embedding_dim:
            raise ValueError(f"Vector dimension mismatch: {len(vectors[0])} != {self.embedding_dim}")

        total = n

        for i in range(0, total, batch_size):
            pts = [
                # consider using the list(vec) to ensure it's a list, not another sequence type
                PointStruct(id=i_id, vector=vec, payload=pl)
                for i_id, vec, pl in zip(
                    ids[i:i + batch_size],
                    vectors[i:i + batch_size],
                    payloads[i:i + batch_size]
                )
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=pts,
                wait=wait,
            )
            done = min(i + batch_size, total)
            if done % 4000 == 0 or done == total:
                logger.info("Upserted %d/%d points", done, total)

    @staticmethod
    def _to_filter(filter_dict: Optional[dict[str, Any]]) -> Optional[Filter]:
        """
        Convert a simple dict to a Qdrant Filter.
        Args:
            filter_dict: dict of field names to values or list of values.
        Supports exact matches and list of values (MatchAny).
        """
        if not filter_dict:
            return None
        must = []
        for k, v in filter_dict.items():
            if isinstance(v, (list, tuple, set)):
                must.append(FieldCondition(key=k, match=MatchAny(any=list(v))))
            else:
                must.append(FieldCondition(key=k, match=MatchValue(value=v)))
        return Filter(must=must) if must else None

    def search(
            self,
            vector: Vector,
            top_k=5,
            filter_dict: Optional[dict[str, Any]] = None,
            score_threshold: Optional[float] = None,
            hnsw_ef: Optional[int] = None,
            with_vectors: bool = False,
    ) -> List[Tuple[IdLike, dict[str, Any], float]]:
        """
        The score is higher = better (cosine/dot) or lower = better (euclidean).
            Args:
                vector: query vector
                top_k: number of results to return
                filter_dict: optional filter dict (exact matches or list of values)
                score_threshold: optional threshold to filter results by score
                hnsw_ef: optional HNSW ef parameter for search (higher = more accurate, default 128)
                with_vectors: whether to include the matched vectors in the results
        Returns: a list of (id, payload, score) tuples.
        """
        q_filter = self._to_filter(filter_dict)
        params = SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=q_filter,
            limit=top_k,
            score_threshold=score_threshold,
            search_params=params,
            with_payload=True,
            with_vectors=with_vectors,
        )
        logger.debug("Search returned %d hits from collection '%s'", len(hits), self.collection_name)
        return [(hit.id, hit.payload, hit.score) for hit in hits]

    def get_by_ids(self, ids: Sequence[IdLike], with_vectors: bool = False) -> List[dict[str, Any]]:
        """
        Retrieve points by their IDs.
            Args:
                ids: list of point IDs to retrieve
                with_vectors: whether to include the vectors in the results
        Returns: a list of (id, payload, vector) tuples. Vector is None if with_vectors is False.
        """
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=with_vectors,
        )
        # Maintain the order of the input IDs
        by_id = {p.id: p.payload for p in points}
        logger.debug("Retrieved %d points by IDs from collection '%s'", len(points), self.collection_name)
        return [by_id.get(i, {}) for i in ids]

    def delete_by_filter(self, filter_dict: dict[str, Any]) -> None:
        """
        Delete points matching the filter.
            Args:
                filter_dict: filter dict (exact matches or list of values)
        """
        q_filter = self._to_filter(filter_dict)
        if not q_filter:
            raise ValueError("Filter dictionary is empty or invalid")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=q_filter,
            wait=True,
        )
        logger.info("Deleted points matching filter in collection '%s'", self.collection_name)

    def count(self, filter_dict: Optional[dict[str, Any]] = None, exact: bool = False) -> int:
        """
        Count points in the collection, optionally filtered.
            Args:
                filter_dict: optional filter dict (exact matches or list of values)
                exact: whether to get an exact count (may be slower)
        Returns: the count of points matching the filter.
        """
        q_filter = self._to_filter(filter_dict)
        stats = self.client.count(
            collection_name=self.collection_name,
            count_filter=q_filter,
            exact=exact,
        )
        logger.debug("Counted %d points in collection '%s'", stats.count, self.collection_name)
        return stats.count

    def clear(self, wait: bool = True) -> None:
        """
        Delete all points in the collection.
            Args:
                wait: whether to wait for the operation to complete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=None,
            wait=wait,
        )
        logger.info("Cleared all points in collection '%s'", self.collection_name)

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection information.

        Returns:
            Dictionary with collection info
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
            "status": info.status,
        }
