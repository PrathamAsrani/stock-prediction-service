import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Extract and process trading books and educational PDFs"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path} with PyPDF2: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for tables)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path} with pdfplumber: {e}")
            return ""
    
    def process_pdf(self, pdf_path: str) -> str:
        """Process a single PDF file"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Try pdfplumber first (better quality)
        text = self.extract_text_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text or len(text) < 100:
            logger.warning(f"pdfplumber extraction poor, trying PyPDF2 for {pdf_path}")
            text = self.extract_text_pypdf2(pdf_path)
        
        if not text:
            logger.error(f"Failed to extract text from {pdf_path}")
            return ""
        
        # Clean text
        text = self._clean_text(text)
        
        return text
    
    def process_directory(self, directory: str) -> List[Document]:
        """Process all PDFs in a directory"""
        pdf_dir = Path(directory)
        documents = []
        
        if not pdf_dir.exists():
            logger.error(f"Directory does not exist: {directory}")
            return documents
        
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_file in pdf_files:
            text = self.process_pdf(str(pdf_file))
            
            if text:
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_file),
                        "filename": pdf_file.name,
                        "type": "book",
                        "category": self._categorize_book(pdf_file.name)
                    }
                )
                documents.append(doc)
        
        logger.info(f"Processed {len(documents)} documents from {directory}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        return chunked_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep important ones
        text = text.replace("\x00", "")
        
        return text
    
    def _categorize_book(self, filename: str) -> str:
        """Categorize book based on filename"""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['technical', 'chart', 'pattern']):
            return 'technical_analysis'
        elif any(word in filename_lower for word in ['fundamental', 'value', 'buffett']):
            return 'fundamental_analysis'
        elif any(word in filename_lower for word in ['swing', 'trade', 'strategy']):
            return 'trading_strategy'
        elif any(word in filename_lower for word in ['psychology', 'emotion', 'discipline']):
            return 'trading_psychology'
        else:
            return 'general'
        