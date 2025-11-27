import io
import csv
import json
from pypdf import PdfReader
from docx import Document
from lxml import etree

def parse_pdf(content: bytes) -> str:
    """Parses a PDF file from bytes and returns its text content."""
    reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_docx(content: bytes) -> str:
    """Parses a DOCX file from bytes and returns its text content."""
    document = Document(io.BytesIO(content))
    text = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(text)

def parse_json(content: str) -> dict:
    """Parses JSON content from a string and returns a dictionary."""
    return json.loads(content)

def parse_xml(content: str) -> str:
    """Parses XML content from a string and returns a pretty-printed string."""
    root = etree.fromstring(content)
    return etree.tostring(root, pretty_print=True, encoding='unicode')

def parse_csv(content: str) -> list[list[str]]:
    """Parses CSV content from a string and returns a list of lists."""
    f = io.StringIO(content)
    reader = csv.reader(f)
    return list(reader)
