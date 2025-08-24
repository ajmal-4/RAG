from typing import Literal, Union

from docling.document_converter import DocumentConverter

from app.core.config import settings


class DoclingService:
    def __init__(self):
        self.converter = DocumentConverter()
        self.page_seperator = settings.docling_page_seperator

    def extract_text_with_docling(
        self, file_path: str, response_format: Literal["markdown", "json"] = "markdown"
    ) -> Union[str, dict]:
        doc = self.converter.convert(file_path).document

        try:
            if response_format == "markdown":
                return doc.export_to_markdown(
                    page_break_placeholder=self.page_seperator
                )
            else:
                return doc.export_to_dict()
        except Exception:
            return doc.export_to_text()

    def map_markdown_to_pages(self, markdown_text: str):
        page_content_mapping = []
        for i, content in enumerate(markdown_text.split(self.page_seperator), start=1):
            page_content_mapping.append({"page": i, "text": content})
        
        return page_content_mapping
