import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)

class ReportsCollector:
    """Collect company annual and quarterly reports"""
    
    def __init__(self, save_path: str = "data/reports"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def download_bse_report(
        self,
        company_code: str,
        report_type: str = "annual"
    ) -> Optional[str]:
        """
        Download report from BSE website
        
        Args:
            company_code: BSE company code
            report_type: 'annual' or 'quarterly'
        """
        try:
            # BSE Corporate Announcements page
            url = f"https://www.bseindia.com/corporates/ann.aspx?scrip={company_code}"
            
            logger.info(f"Fetching reports for BSE code: {company_code}")
            
            # This is a placeholder - actual implementation would need to:
            # 1. Navigate to BSE website
            # 2. Search for company
            # 3. Find annual report links
            # 4. Download PDF
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading BSE report: {str(e)}")
            return None
    
    def download_nse_report(
        self,
        symbol: str,
        report_type: str = "annual"
    ) -> Optional[str]:
        """Download report from NSE website"""
        try:
            # NSE does not provide direct report downloads
            # Reports are usually on company websites
            logger.info(f"NSE reports typically found on company websites for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading NSE report: {str(e)}")
            return None
    
    def download_from_company_website(
        self,
        company_url: str,
        symbol: str
    ) -> List[str]:
        """
        Download reports from company's investor relations page
        
        This is a template - needs customization per company
        """
        try:
            response = requests.get(company_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find PDF links (customize based on website structure)
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf') and any(
                    keyword in href.lower() 
                    for keyword in ['annual', 'report', 'quarterly']
                ):
                    pdf_links.append(href)
            
            # Download PDFs
            downloaded = []
            for pdf_url in pdf_links[:5]:  # Limit to 5
                if not pdf_url.startswith('http'):
                    pdf_url = company_url.rstrip('/') + '/' + pdf_url.lstrip('/')
                
                filename = self._download_pdf(pdf_url, symbol)
                if filename:
                    downloaded.append(filename)
                
                time.sleep(1)  # Be respectful
            
            return downloaded
            
        except Exception as e:
            logger.error(f"Error downloading from company website: {str(e)}")
            return []
    
    def _download_pdf(self, url: str, symbol: str) -> Optional[str]:
        """Download a PDF file"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                # Generate filename
                timestamp = int(time.time())
                filename = f"{symbol}_{timestamp}.pdf"
                filepath = self.save_path / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded: {filename}")
                return str(filepath)
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return None
    
    def get_report_metadata(self, symbol: str) -> Dict:
        """Get metadata about available reports"""
        
        reports_dir = self.save_path
        symbol_reports = list(reports_dir.glob(f"{symbol}_*.pdf"))
        
        return {
            'symbol': symbol,
            'total_reports': len(symbol_reports),
            'reports': [r.name for r in symbol_reports]
        }
    