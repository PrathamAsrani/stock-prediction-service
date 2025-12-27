import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from langchain.schema import Document
import logging
import re

logger = logging.getLogger(__name__)

class ReportProcessor:
    """Process annual and quarterly company reports"""
    
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'profit', 'ebitda', 'eps', 'roe', 'roce',
            'debt', 'equity', 'cash flow', 'dividend', 'margin',
            'sales', 'assets', 'liabilities', 'reserves'
        ]
    
    def process_annual_report(self, report_path: str, company: str, year: int) -> List[Document]:
        """Process an annual report PDF"""
        from app.knowledge_base.pdf_processor import PDFProcessor
        
        pdf_processor = PDFProcessor(chunk_size=1500, chunk_overlap=300)
        text = pdf_processor.process_pdf(report_path)
        
        if not text:
            return []
        
        # Extract key sections
        sections = self._extract_report_sections(text)
        
        documents = []
        for section_name, section_text in sections.items():
            # Extract financial metrics
            metrics = self._extract_financial_metrics(section_text)
            
            doc = Document(
                page_content=section_text,
                metadata={
                    "source": report_path,
                    "company": company,
                    "year": year,
                    "type": "annual_report",
                    "section": section_name,
                    "metrics": metrics
                }
            )
            documents.append(doc)
        
        logger.info(f"Processed annual report for {company} ({year}): {len(documents)} sections")
        
        return documents
    
    def process_quarterly_report(self, report_path: str, company: str, quarter: str, year: int) -> List[Document]:
        """Process a quarterly report"""
        from app.knowledge_base.pdf_processor import PDFProcessor
        
        pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
        text = pdf_processor.process_pdf(report_path)
        
        if not text:
            return []
        
        # Extract key sections
        sections = self._extract_report_sections(text)
        
        documents = []
        for section_name, section_text in sections.items():
            metrics = self._extract_financial_metrics(section_text)
            
            doc = Document(
                page_content=section_text,
                metadata={
                    "source": report_path,
                    "company": company,
                    "year": year,
                    "quarter": quarter,
                    "type": "quarterly_report",
                    "section": section_name,
                    "metrics": metrics
                }
            )
            documents.append(doc)
        
        logger.info(f"Processed quarterly report for {company} (Q{quarter} {year}): {len(documents)} sections")
        
        return documents
    
    def _extract_report_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from report"""
        sections = {}
        
        # Common section headers
        section_headers = {
            'management_discussion': [
                'management discussion', 'md&a', 'management commentary',
                'directors report', 'management report'
            ],
            'financial_statements': [
                'financial statements', 'balance sheet', 'income statement',
                'profit and loss', 'cash flow statement'
            ],
            'business_overview': [
                'business overview', 'company overview', 'business profile',
                'about the company', 'business operations'
            ],
            'risk_factors': [
                'risk factors', 'risks', 'risk management'
            ],
            'future_outlook': [
                'future outlook', 'outlook', 'forward looking',
                'future prospects', 'guidance'
            ]
        }
        
        # Split text into potential sections
        lines = text.split('\n')
        current_section = 'general'
        current_text = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            section_found = False
            for section_key, headers in section_headers.items():
                if any(header in line_lower for header in headers):
                    # Save previous section
                    if current_text:
                        sections[current_section] = '\n'.join(current_text)
                    
                    # Start new section
                    current_section = section_key
                    current_text = []
                    section_found = True
                    break
            
            if not section_found:
                current_text.append(line)
        
        # Save last section
        if current_text:
            sections[current_section] = '\n'.join(current_text)
        
        return sections
    
    def _extract_financial_metrics(self, text: str) -> Dict[str, float]:
        """Extract financial metrics from text"""
        metrics = {}
        
        # Patterns for financial numbers
        patterns = {
            'revenue': r'(?:revenue|sales|turnover).*?(?:rs\.?|₹|inr)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|cr|million|mn|billion|bn)',
            'profit': r'(?:profit|net income|earnings).*?(?:rs\.?|₹|inr)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|cr|million|mn)',
            'eps': r'(?:eps|earnings per share).*?(?:rs\.?|₹|inr)\s*(\d+(?:\.\d+)?)',
            'dividend': r'(?:dividend).*?(?:rs\.?|₹|inr)\s*(\d+(?:\.\d+)?)',
            'roe': r'(?:roe|return on equity).*?(\d+(?:\.\d+)?)\s*%',
            'debt': r'(?:debt|borrowings).*?(?:rs\.?|₹|inr)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|cr|million|mn)'
        }
        
        text_lower = text.lower()
        
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    # Clean and convert number
                    value = matches[0].replace(',', '')
                    metrics[metric] = float(value)
                except (ValueError, IndexError):
                    continue
        
        return metrics
    
    def process_reports_directory(self, directory: str, report_type: str = 'annual') -> List[Document]:
        """Process all reports in a directory"""
        reports_dir = Path(directory)
        documents = []
        
        if not reports_dir.exists():
            logger.error(f"Reports directory does not exist: {directory}")
            return documents
        
        # Expected naming: COMPANY_YEAR_Q[1-4].pdf or COMPANY_YEAR_ANNUAL.pdf
        pattern = "*.pdf"
        report_files = list(reports_dir.glob(pattern))
        
        logger.info(f"Found {len(report_files)} {report_type} reports in {directory}")
        
        for report_file in report_files:
            try:
                # Parse filename
                filename = report_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    company = parts[0]
                    year = int(parts[1])
                    
                    if report_type == 'quarterly' and len(parts) >= 3:
                        quarter = parts[2].replace('Q', '')
                        docs = self.process_quarterly_report(str(report_file), company, quarter, year)
                    else:
                        docs = self.process_annual_report(str(report_file), company, year)
                    
                    documents.extend(docs)
            
            except Exception as e:
                logger.error(f"Error processing report {report_file}: {e}")
                continue
        
        logger.info(f"Processed {len(documents)} document sections from {report_type} reports")
        
        return documents
    