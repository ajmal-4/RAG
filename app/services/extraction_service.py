from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.core.config import settings
from app.core.embedding import get_embeddings
from app.services.docling_service import DoclingService


class ExtractionService:
    def __init__(self):
        self.is_docling_retriever = settings.is_docling_retriever
        self.docling_service = DoclingService()
        self.custom_splitter = settings.custom_splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_split_size,
            chunk_overlap=settings.chunk_split_overlap,
        )
        self.qdrant_client = QdrantClient(url=settings.qdrant_url)
        self.embeddings = get_embeddings()
        self._load_vector_db_creds()

    def _load_vector_db_creds(self):
        """Load credentials for allowed vector databases from settings."""
        db_mapping = {
            "qdrant": {
                "url_attr": "qdrant_url",
                "collection_attr": "qdrant_collection_name",
                "url_value": settings.qdrant_url,
                "collection_value": settings.qdrant_collection_name,
            },
            "weaviate": {
                # Add weaviate credentials here if needed
            },
        }

        for v_db in settings.allowed_vector_db:
            creds = db_mapping.get(v_db)
            if creds:
                setattr(self, creds["url_attr"], creds["url_value"])
                setattr(self, creds["collection_attr"], creds["collection_value"])

    def extract_using_docling(self, file_path: str):
        response_format = settings.docling_response_format

        # Extract full content from the file
        full_content = self.docling_service.extract_text_with_docling(
            file_path, response_format
        )

        if response_format == "markdown":
            # Returns content correponding to the pages
            page_content_mapping = self.docling_service.map_markdown_to_pages(
                full_content
            )
        else:
            pass

        return page_content_mapping
    
    def upsert_to_qdrant(self, documents: Document, vectors: list[list[float]]):

        points = []
        for doc, vec in zip(documents, vectors):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vec,
                    payload={
                        "metadata": doc.metadata,
                        "page_content": doc.page_content,
                    },
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.qdrant_collection_name,
            points=points
        )

        print(f"Upserted {len(points)} chunks")
        return len(points)

    def chunk_and_upsert_documents(
        self, extracted_contents: list[dict], file_metadata: dict
    ):
        if self.custom_splitter:
            # Use custom splitting strategy
            pass

        else:
            # Use RecursiveCharacterTextSplitter of langchain
            all_docs = []
            for page in extracted_contents:
                # Split each page
                page_chunks = self.splitter.create_documents([page["text"]])

                # Attach metadata: source, page, chunk_index
                for i, doc in enumerate(page_chunks, 1):
                    doc.metadata.update(
                        {
                            "source": file_metadata["file_name"],
                            "page": page["page"],
                            "chunk_index": i,
                        }
                    )
                all_docs.extend(page_chunks)
        
        # -------- Batch Embedding --------
        texts = [doc.page_content for doc in all_docs]
        BATCH_SIZE = settings.embedding_batch_size

        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_vecs = self.embeddings.embed_documents(batch)
            vectors.extend(batch_vecs)
        
        upserted_chunks = self.upsert_to_qdrant(all_docs, vectors)
        return upserted_chunks

    def extract_chunk_upsert_document(self, file_path: str, file_metadata: dict):
        if self.is_docling_retriever:
            extracted_contents = self.extract_using_docling(file_path)

            upserted_chunks = self.chunk_and_upsert_documents(
                extracted_contents, file_metadata
            )

        else:
            # Use any other parser later.
            pass

        return upserted_chunks
