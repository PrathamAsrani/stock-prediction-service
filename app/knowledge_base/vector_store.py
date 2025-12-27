import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from langchain.schema import Document
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage vector embeddings and similarity search using ChromaDB"""
    
    def __init__(
        self, 
        persist_directory: str = "knowledge_base_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.persist_directory)
        ))
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create or get collections
        self.collections = {
            'books': self._get_or_create_collection('trading_books'),
            'reports': self._get_or_create_collection('company_reports'),
            'patterns': self._get_or_create_collection('chart_patterns'),
            'general': self._get_or_create_collection('general_knowledge')
        }
        
        logger.info("VectorStore initialized successfully")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=name)
            logger.info(f"Loaded existing collection: {name}")
        except:
            collection = self.client.create_collection(name=name)
            logger.info(f"Created new collection: {name}")
        
        return collection
    
    def add_documents(
        self, 
        documents: List[Document], 
        collection_type: str = 'general'
    ):
        """Add documents to the vector store"""
        
        if collection_type not in self.collections:
            logger.error(f"Invalid collection type: {collection_type}")
            return
        
        collection = self.collections[collection_type]
        
        logger.info(f"Adding {len(documents)} documents to {collection_type} collection")
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{collection_type}_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            collection.add(
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        logger.info(f"Successfully added {len(documents)} documents to {collection_type}")
    
    def search(
        self, 
        query: str, 
        collection_type: str = 'general',
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents"""
        
        if collection_type not in self.collections:
            logger.error(f"Invalid collection type: {collection_type}")
            return []
        
        collection = self.collections[collection_type]
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def search_all_collections(
        self, 
        query: str, 
        n_results_per_collection: int = 3
    ) -> Dict[str, List[Dict]]:
        """Search across all collections"""
        
        all_results = {}
        
        for collection_name in self.collections.keys():
            results = self.search(
                query=query,
                collection_type=collection_name,
                n_results=n_results_per_collection
            )
            all_results[collection_name] = results
        
        return all_results
    
    def get_relevant_context(
        self,
        query: str,
        symbol: Optional[str] = None,
        max_tokens: int = 2000
    ) -> str:
        """Get relevant context for a query, optimized for token limits"""
        
        # Search all collections
        all_results = self.search_all_collections(query, n_results_per_collection=2)
        
        # If symbol is provided, also search for company-specific info
        if symbol:
            company_results = self.search(
                query=f"company analysis {symbol}",
                collection_type='reports',
                n_results=3,
                filter_metadata={'company': symbol}
            )
            all_results['company_specific'] = company_results
        
        # Compile context
        context_parts = []
        current_tokens = 0
        
        for collection_name, results in all_results.items():
            for result in results:
                text = result['document']
                # Rough token estimation (4 chars ≈ 1 token)
                estimated_tokens = len(text) // 4
                
                if current_tokens + estimated_tokens > max_tokens:
                    break
                
                context_parts.append(f"[{collection_name.upper()}] {text}")
                current_tokens += estimated_tokens
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Compiled context with ~{current_tokens} tokens from {len(context_parts)} sources")
        
        return context
    
    def delete_collection(self, collection_type: str):
        """Delete a collection"""
        if collection_type in self.collections:
            self.client.delete_collection(name=collection_type)
            logger.info(f"Deleted collection: {collection_type}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about all collections"""
        stats = {}
        
        for name, collection in self.collections.items():
            count = collection.count()
            stats[name] = {
                'document_count': count
            }
        
        return stats
    
    def persist(self):
        """Persist the vector store to disk"""
        self.client.persist()
        logger.info("Vector store persisted to disk")
        