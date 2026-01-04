import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pathlib import Path
import logging
from tqdm import tqdm

from app.knowledge_base.pdf_processor import PDFProcessor
from app.knowledge_base.report_processor import ReportProcessor
from app.knowledge_base.pattern_extractor import PatternExtractor
from app.knowledge_base.vector_store import VectorStore
from app.services.data_service import DataService
from app.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBaseBuilder:
    """Build comprehensive knowledge base from all sources"""
    
    def __init__(self):
        self.vector_store = VectorStore(
            persist_directory=settings.KNOWLEDGE_BASE_PATH,
            embedding_model=settings.EMBEDDING_MODEL
        )
        self.pdf_processor = PDFProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.report_processor = ReportProcessor()
        self.pattern_extractor = PatternExtractor()
        self.data_service = DataService()
    
    async def build_books_index(self):
        """Index trading books and educational PDFs"""
        logger.info("="*60)
        logger.info("INDEXING TRADING BOOKS")
        logger.info("="*60)
        
        books_path = Path(settings.BOOKS_PATH)
        
        if not books_path.exists():
            logger.warning(f"Books directory does not exist: {books_path}")
            logger.info("Creating directory and downloading sample books...")
            books_path.mkdir(parents=True, exist_ok=True)
            await self._download_sample_books(books_path)
        
        # Process all PDFs
        documents = self.pdf_processor.process_directory(str(books_path))
        
        if documents:
            # Chunk documents
            chunked_docs = self.pdf_processor.chunk_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(chunked_docs, collection_type='books')
            
            logger.info(f"✓ Indexed {len(chunked_docs)} chunks from {len(documents)} books")
        else:
            logger.warning("No books found to index")
    
    async def build_reports_index(self):
        """Index company annual and quarterly reports"""
        logger.info("="*60)
        logger.info("INDEXING COMPANY REPORTS")
        logger.info("="*60)
        
        reports_path = Path(settings.REPORTS_PATH)
        
        if not reports_path.exists():
            logger.warning(f"Reports directory does not exist: {reports_path}")
            logger.info("Creating directory...")
            reports_path.mkdir(parents=True, exist_ok=True)
            # In production, download from company websites or APIs
            return
        
        # Process annual reports
        annual_path = reports_path / "annual"
        if annual_path.exists():
            annual_docs = self.report_processor.process_reports_directory(
                str(annual_path),
                report_type='annual'
            )
            if annual_docs:
                self.vector_store.add_documents(annual_docs, collection_type='reports')
                logger.info(f"✓ Indexed {len(annual_docs)} annual report sections")
        
        # Process quarterly reports
        quarterly_path = reports_path / "quarterly"
        if quarterly_path.exists():
            quarterly_docs = self.report_processor.process_reports_directory(
                str(quarterly_path),
                report_type='quarterly'
            )
            if quarterly_docs:
                self.vector_store.add_documents(quarterly_docs, collection_type='reports')
                logger.info(f"✓ Indexed {len(quarterly_docs)} quarterly report sections")
    
    async def build_patterns_index(self, symbols: list):
        """Extract and index historical chart patterns"""
        logger.info("="*60)
        logger.info("EXTRACTING CHART PATTERNS")
        logger.info("="*60)
        
        all_pattern_docs = []
        
        for symbol in tqdm(symbols, desc="Processing symbols"):
            try:
                # Fetch historical data
                candles = self.data_service.fetch_historical_data(
                    symbol=symbol,
                    exchange="NSE",
                    days=365 * 5  # 5 years of data
                )
                
                if not candles:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = self.data_service.candles_to_dataframe(candles)
                
                # Extract patterns
                pattern_docs = self.pattern_extractor.extract_patterns_from_history(
                    df, symbol
                )
                
                all_pattern_docs.extend(pattern_docs)
                logger.info(f"✓ Extracted {len(pattern_docs)} patterns from {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        if all_pattern_docs:
            self.vector_store.add_documents(all_pattern_docs, collection_type='patterns')
            logger.info(f"✓ Indexed {len(all_pattern_docs)} total patterns")
    
    async def _download_sample_books(self, books_path: Path):
        """Download sample trading books (placeholder)"""
        logger.info("Sample books would be downloaded here...")
        logger.info("For production:")
        logger.info("1. Add your trading books (PDFs) to: " + str(books_path))
        logger.info("2. Recommended books:")
        logger.info("   - Technical Analysis of the Financial Markets by John Murphy")
        logger.info("   - Trading in the Zone by Mark Douglas")
        logger.info("   - How to Make Money in Stocks by William O'Neil")
        logger.info("   - The New Trading for a Living by Dr. Alexander Elder")
    
    async def build_full_index(self, symbols: list):
        """Build complete knowledge base"""
        logger.info("\n" + "="*60)
        logger.info("BUILDING COMPLETE KNOWLEDGE BASE")
        logger.info("="*60 + "\n")
        
        # Build books index
        await self.build_books_index()
        
        # Build reports index
        await self.build_reports_index()
        
        # Build patterns index
        await self.build_patterns_index(symbols)
        
        # Persist vector store
        self.vector_store.persist()
        
        # Show statistics
        stats = self.vector_store.get_collection_stats()
        
        logger.info("\n" + "="*60)
        logger.info("KNOWLEDGE BASE STATISTICS")
        logger.info("="*60)
        for collection, info in stats.items():
            logger.info(f"{collection.upper()}: {info['document_count']} documents")
        logger.info("="*60 + "\n")
        
        logger.info("✓ Knowledge base built successfully!")

async def main():
    """Main function"""
    
    # List of stocks to analyze (can be extended)
    # symbols = [
    #     'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
    #     'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC',
    #     'BAJFINANCE', 'LT', 'ASIANPAINT', 'AXISBANK', 'MARUTI',
    #     'ADANIPORTS', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO'
    # ]
    symbols = [
        'ITC',    # Has fallback data
        'PGHL',   # Has fallback data
    ]
    
    builder = KnowledgeBaseBuilder()
    
    try:
        await builder.build_full_index(symbols)
        logger.info("\n✓ KNOWLEDGE BASE BUILD COMPLETE!")
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
    