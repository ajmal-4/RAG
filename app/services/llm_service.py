from typing import List, Dict, Any, AsyncGenerator

import numpy as np
from sklearn.cluster import KMeans
from langchain.messages import SystemMessage, HumanMessage

from app.core.llm import get_llm, get_model_id
from app.services.llm_utils import load_prompt
from app.schemas.rag_api_schema import ChatRequest
from app.services.vector_db_agent import VectorDbAgent


class LLMService:
    def __init__(self):
        self.vector_db_agent = VectorDbAgent()

    async def generate_simple_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        model_name = request.model_name or "deepseek"
        llm_client = get_llm(model_name)
        model_id = get_model_id(model_name)

        stream = await llm_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": request.question}],
            stream=True,
        )

        full_response = ""

        async for chunk in stream:  # sync generator
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                # yield chunk asynchronously
                yield delta

    async def generate_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        model_name = request.model_name or "qwen"
        llm_client = get_llm(model_name)

        system_prompt = load_prompt("simple_response")["SYSTEM"]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.question)
        ]

        async for token in llm_client.stream(messages):
            yield token

    async def summarize_with_kmeans_clustering(
        self, 
        collection_name, 
        filters,
        n_clusters: int = 5,
        top_k: int = 1,
        return_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        
        # Scroll the complete points associated to the document, from the collection
        points = self.vector_db_agent.scroll_from_qdrant(
            collection_name, filters, with_vector=True
        )
        if not points:
            return []

        # Keep only points that have vectors
        valid_points = [
            p
            for p in points
            if (
                getattr(p, "vector", None)
                if hasattr(p, "vector")
                else p.get("vector", None)
            )
            is not None
        ]
        if not valid_points:
            return []

        # Extract vectors in one go
        vectors = np.array(
            [
                getattr(p, "vector", None) if hasattr(p, "vector") else p.get("vector")
                for p in valid_points
            ],
            dtype=float,
        )

        X = vectors
        n_points = X.shape[0]
        k = min(n_clusters, n_points)

        # If only one cluster, just pick closest to mean
        if k == 1:
            centroid = X.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(X - centroid, axis=1)
            idxs = np.argsort(dists)[: min(top_k, n_points)]
            results = []
            for idx in idxs:
                p = valid_points[idx]
                pid = getattr(p, "id", None) or p.get("id")
                payload = getattr(p, "payload", None) or p.get("payload")
                entry = {"id": pid, "payload": payload, "score": float(dists[idx])}
                if return_vectors:
                    entry["vector"] = X[idx].tolist()
                results.append(entry)
            return results

        # Run KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        results: List[Dict[str, Any]] = []
        for cluster_idx in range(k):
            member_idxs = np.where(labels == cluster_idx)[0]
            if member_idxs.size == 0:
                continue

            cluster_vectors = X[member_idxs]
            centroid = centroids[cluster_idx]
            dists = np.linalg.norm(cluster_vectors - centroid, axis=1)

            sorted_local = np.argsort(dists)
            take = min(top_k, len(sorted_local))

            for local_pos in sorted_local[:take]:
                global_idx = member_idxs[local_pos]
                p = valid_points[global_idx]
                pid = getattr(p, "id", None) or p.get("id")
                payload = getattr(p, "payload", None) or p.get("payload")
                entry = {
                    "id": pid,
                    "payload": payload,
                    "score": float(dists[local_pos]),
                }
                if return_vectors:
                    entry["vector"] = X[global_idx].tolist()
                results.append(entry)

        return results