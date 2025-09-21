from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition

from app.core.config import settings
from app.core.embedding import get_embeddings


class VectorDbAgent:
    """Handles interactions with a vector database and embeddings."""
    def __init__(self):
        self.qdrant_client = QdrantClient(url=settings.qdrant_url)
        self.embeddings = get_embeddings()
    
    def _build_qdrant_filter(
        self,
        filters: Dict[str, Dict[str, Any]]
    ) -> Filter:
        """
        Converts a dictionary of 'must' and 'must_not' conditions into a Qdrant Filter.

        Example input:
        {
            "must": {"source": "Payslip_Apr_2025.pdf", "page": 1},
            "must_not": {"chunk_index": 0}
        }

        Returns:
            Qdrant Filter object 
        """

        must_conditions = []
        must_not_conditions = []

        # Build 'must' conditions
        for key, value in filters.get("must", {}).items():
            must_conditions.append(FieldCondition(key=f"metadata.{key}", match={"value": value}))

        # Build 'must_not' conditions
        for key, value in filters.get("must_not", {}).items():
            must_not_conditions.append(FieldCondition(key=f"metadata.{key}", match={"value": value}))

        return Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )

    def retrieve_from_qdrant(
        self, query, metadata: Optional[dict] = {}, collection_name=""
    ):
        """Retrieve from semantic chunks from vector db"""
        hits = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=self.embeddings.embed_query(query),
            limit=settings.retrieval_chunk_limit,
        )

        retrieved_chunks = [{"chunk": hit.payload, "score": hit.score} for hit in hits]
        return retrieved_chunks
    
    def scroll_from_qdrant(
        self, 
        collection_name, 
        filters: Optional[Dict[str, Any]] = None,
        with_vector=False
    ):
        """Retrieve full chuks according to filters from vector db"""

        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_qdrant_filter(filters)
        
        points, _ = self.qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=with_vector,
            scroll_filter=qdrant_filter,
        )

        return points
    