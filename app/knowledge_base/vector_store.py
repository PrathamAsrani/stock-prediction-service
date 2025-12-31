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
        self.persist_directory.mkdir(exist_ok=True, parents=True)
        
        # Initialize ChromaDB client with new API
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        try:
            # Use the new PersistentClient API
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            # Fallback to EphemeralClient for testing
            logger.warning("Falling back to in-memory EphemeralClient")
            self.client = chromadb.EphemeralClient()
        
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
            logger.info(f"Loaded existing collection: {name} with {collection.count()} documents")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
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
        
        # Get current count to generate unique IDs
        current_count = collection.count()
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{collection_type}_{current_count + i}" for i in range(len(documents))]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            try:
                collection.add(
                    embeddings=embeddings[i:batch_end],
                    documents=texts[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                continue
        
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
        
        # Check if collection is empty
        if collection.count() == 0:
            logger.warning(f"Collection {collection_type} is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                where=filter_metadata if filter_metadata else None
            )
        except Exception as e:
            logger.error(f"Error searching collection {collection_type}: {e}")
            return []
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
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
            try:
                company_results = self.search(
                    query=f"company analysis {symbol}",
                    collection_type='reports',
                    n_results=3,
                    filter_metadata={'company': symbol}
                )
                all_results['company_specific'] = company_results
            except Exception as e:
                logger.warning(f"Could not search company-specific data: {e}")
        
        # Compile context
        context_parts = []
        current_tokens = 0
        
        for collection_name, results in all_results.items():
            if not results:
                continue
                
            for result in results:
                text = result['document']
                # Rough token estimation (4 chars ≈ 1 token)
                estimated_tokens = len(text) // 4
                
                if current_tokens + estimated_tokens > max_tokens:
                    break
                
                context_parts.append(f"[{collection_name.upper()}] {text}")
                current_tokens += estimated_tokens
        
        if context_parts:
            context = "\n\n".join(context_parts)
            logger.info(f"Compiled context with ~{current_tokens} tokens from {len(context_parts)} sources")
        else:
            context = "No relevant context found in knowledge base."
            logger.warning("No context found in knowledge base")
        
        return context
    
    def delete_collection(self, collection_type: str):
        """Delete a collection"""
        if collection_type in self.collections:
            try:
                self.client.delete_collection(name=collection_type)
                logger.info(f"Deleted collection: {collection_type}")
                # Remove from collections dict
                del self.collections[collection_type]
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about all collections"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    'document_count': count
                }
            except Exception as e:
                logger.error(f"Error getting stats for {name}: {e}")
                stats[name] = {
                    'document_count': 0,
                    'error': str(e)
                }
        
        return stats
    
    def reset_all_collections(self):
        """Delete and recreate all collections (useful for development)"""
        logger.warning("Resetting all collections...")
        
        for collection_type in list(self.collections.keys()):
            self.delete_collection(collection_type)
        
        # Recreate collections
        self.collections = {
            'books': self._get_or_create_collection('trading_books'),
            'reports': self._get_or_create_collection('company_reports'),
            'patterns': self._get_or_create_collection('chart_patterns'),
            'general': self._get_or_create_collection('general_knowledge')
        }
        
        logger.info("All collections reset")
        