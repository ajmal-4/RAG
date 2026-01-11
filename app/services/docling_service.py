from typing import Literal, Union

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from app.core.config import settings


class DoclingService:
    def __init__(self):
        # Configure pipeline to handle images
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for image-based PDFs
        pipeline_options.do_table_structure = True  # Extract tables properly

        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        # Initialize converter with OCR enabled
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
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
