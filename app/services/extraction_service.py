from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

from app.core.config import settings
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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embed_device},
        )
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

        # Upsert all at once to Qdrant - Currently only qdrant
        _ = Qdrant.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            url=self.qdrant_url,
            collection_name=self.qdrant_collection_name,
        )

        print(f"Appended {len(all_docs)} no.of chunks")
        return len(all_docs)

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
