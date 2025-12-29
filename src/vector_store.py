"""
Vector Store Module
Manages vector database operations for storing and retrieving embeddings.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not available")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not available")


class VectorStore:
    """Vector database interface for storing and querying embeddings."""
    
    def __init__(self,
                 db_type: str = "chroma",
                 collection_name: str = "arxiv_cs_papers",
                 persist_directory: Optional[str] = None,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        self.db_type = db_type.lower()
        self.collection_name = collection_name
        
        if self.db_type == "chroma":
            if not CHROMADB_AVAILABLE:
                raise ImportError("chromadb is required for ChromaDB vector store")
            self._init_chroma(persist_directory)
        elif self.db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError("qdrant-client is required for Qdrant vector store")
            self._init_qdrant(qdrant_host, qdrant_port)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")
    
    def _init_chroma(self, persist_directory: Optional[str]):
        """Initialize ChromaDB."""
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    def _init_qdrant(self, host: str, port: int):
        """Initialize Qdrant."""
        self.client = QdrantClient(host=host, port=port)
        
        # Check if collection exists, create if not
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if not collection_exists:
            # We'll create it when we know the vector size
            self.collection_created = False
            logger.info(f"Qdrant collection {self.collection_name} will be created on first insert")
        else:
            self.collection_created = True
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")
    
    def add_chunks(self, chunks: List[Dict[str, any]], embeddings: Optional[List] = None):
        """
        Add chunks to vector store.
        
        Args:
            chunks: List of chunks with 'text', 'metadata', and optionally 'embedding'
            embeddings: Optional pre-computed embeddings (if not in chunks)
        """
        if not chunks:
            return
        
        # Extract embeddings
        if embeddings is None:
            embeddings = [chunk.get('embedding') for chunk in chunks]
            if any(emb is None for emb in embeddings):
                raise ValueError("Embeddings must be provided either in chunks or as separate parameter")
        
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in chunks]
        
        # Generate IDs
        ids = [f"{meta.get('paper_id', 'unknown')}_chunk_{meta.get('chunk_index', i)}" 
               for i, meta in enumerate(metadatas)]
        
        if self.db_type == "chroma":
            self._add_chroma(ids, texts, embeddings, metadatas)
        elif self.db_type == "qdrant":
            self._add_qdrant(ids, texts, embeddings, metadatas)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def _add_chroma(self, ids: List[str], texts: List[str], embeddings: List, metadatas: List[Dict]):
        """Add to ChromaDB."""
        # Convert embeddings to list format if needed
        embeddings_list = []
        for emb in embeddings:
            if isinstance(emb, list):
                embeddings_list.append(emb)
            else:
                embeddings_list.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))
        
        # ChromaDB expects metadata values to be strings, numbers, or bools
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_meta[k] = v
                else:
                    cleaned_meta[k] = str(v)
            cleaned_metadatas.append(cleaned_meta)
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=cleaned_metadatas
        )
    
    def _add_qdrant(self, ids: List[str], texts: List[str], embeddings: List, metadatas: List[Dict]):
        """Add to Qdrant."""
        # Get embedding dimension
        first_emb = embeddings[0]
        if isinstance(first_emb, list):
            vector_size = len(first_emb)
        else:
            vector_size = first_emb.shape[0] if hasattr(first_emb, 'shape') else len(first_emb)
        
        # Create collection if needed
        if not self.collection_created:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            self.collection_created = True
            logger.info(f"Created Qdrant collection {self.collection_name} with vector size {vector_size}")
        
        # Prepare points
        points = []
        for i, (id_val, text, emb, meta) in enumerate(zip(ids, texts, embeddings, metadatas)):
            # Convert embedding to list
            if isinstance(emb, list):
                vector = emb
            else:
                vector = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            
            # Qdrant payload (metadata)
            payload = {
                'text': text,
                **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            }
            
            points.append(PointStruct(
                id=i,  # Qdrant uses integer IDs
                vector=vector,
                payload=payload
            ))
        
        # Batch insert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self,
               query_embedding: List[float],
               top_k: int = 10,
               filter_metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with 'text', 'metadata', 'score'
        """
        if self.db_type == "chroma":
            return self._search_chroma(query_embedding, top_k, filter_metadata)
        elif self.db_type == "qdrant":
            return self._search_qdrant(query_embedding, top_k, filter_metadata)
    
    def _search_chroma(self, query_embedding: List[float], top_k: int, filter_metadata: Optional[Dict]) -> List[Dict]:
        """Search in ChromaDB."""
        where = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def _search_qdrant(self, query_embedding: List[float], top_k: int, filter_metadata: Optional[Dict]) -> List[Dict]:
        """Search in Qdrant."""
        # Convert filter to Qdrant format
        query_filter = None
        if filter_metadata:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'text': result.payload.get('text', ''),
                'metadata': {k: v for k, v in result.payload.items() if k != 'text'},
                'score': result.score
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.db_type == "chroma":
            count = self.collection.count()
            return {'total_chunks': count, 'collection_name': self.collection_name}
        elif self.db_type == "qdrant":
            info = self.client.get_collection(self.collection_name)
            return {
                'total_chunks': info.points_count,
                'collection_name': self.collection_name,
                'vector_size': info.config.params.vectors.size
            }

