"""
Embedding generation module using OpenAI API.
Handles creation and storage of embeddings for semantic search.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Embedding model to use.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_dimension = 1536  # For text-embedding-3-small
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the embedding.
        """
        try:
            # Truncate text to avoid token limits
            text = text[:8000]
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embeddings.
        """
        try:
            # Truncate texts to avoid token limits
            texts = [text[:8000] for text in texts]
            
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Sort by index to maintain order
            embeddings = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in embeddings]
            
            logger.debug(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise


class EmbeddingStore:
    """Stores and retrieves embeddings from PostgreSQL with pgvector."""
    
    def __init__(self, db_connection):
        """
        Initialize embedding store.
        
        Args:
            db_connection: PostgreSQL connection object.
        """
        self.db = db_connection
        self.embedding_generator = EmbeddingGenerator()
    
    def store_embedding(
        self,
        project_id: str,
        content_type: str,
        content_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Generate and store embedding for content.
        
        Args:
            project_id: Project identifier.
            content_type: Type of content ('summary', 'decision', 'blocker', etc.).
            content_id: Unique identifier for the content.
            text: Text content to embed.
            metadata: Optional metadata to store with embedding.
            
        Returns:
            ID of the stored embedding.
        """
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(text)
            
            # Convert embedding to string format for pgvector
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata_json = json.dumps(metadata)
            
            # Store in database
            cursor = self.db.cursor()
            
            insert_sql = """
            INSERT INTO project_embeddings 
            (project_id, content_type, content_id, embedding, metadata)
            VALUES (%s, %s, %s, %s::vector, %s::jsonb)
            RETURNING id;
            """
            
            cursor.execute(insert_sql, (
                project_id,
                content_type,
                content_id,
                embedding_str,
                metadata_json
            ))
            
            embedding_id = cursor.fetchone()[0]
            self.db.commit()
            
            logger.info(f"Stored embedding {embedding_id} for {content_type} {content_id}")
            return embedding_id
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to store embedding: {e}")
            raise
    
    def store_embeddings_batch(
        self,
        project_id: str,
        content_type: str,
        items: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Store multiple embeddings in batch.
        
        Args:
            project_id: Project identifier.
            content_type: Type of content.
            items: List of dicts with 'content_id', 'text', and optional 'metadata'.
            
        Returns:
            List of stored embedding IDs.
        """
        try:
            # Extract texts for batch embedding
            texts = [item["text"] for item in items]
            
            # Generate embeddings in batch
            embeddings = self.embedding_generator.generate_embeddings_batch(texts)
            
            # Store all embeddings
            cursor = self.db.cursor()
            embedding_ids = []
            
            for i, item in enumerate(items):
                embedding_str = "[" + ",".join(str(x) for x in embeddings[i]) + "]"
                metadata_json = json.dumps(item.get("metadata", {}))
                
                insert_sql = """
                INSERT INTO project_embeddings 
                (project_id, content_type, content_id, embedding, metadata)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                RETURNING id;
                """
                
                cursor.execute(insert_sql, (
                    project_id,
                    content_type,
                    item["content_id"],
                    embedding_str,
                    metadata_json
                ))
                
                embedding_id = cursor.fetchone()[0]
                embedding_ids.append(embedding_id)
            
            self.db.commit()
            logger.info(f"Stored {len(embedding_ids)} embeddings in batch")
            return embedding_ids
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to store batch embeddings: {e}")
            raise
    
    def semantic_search(
        self,
        project_id: str,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            project_id: Project identifier.
            query: Query text to search for.
            content_type: Optional filter by content type.
            limit: Maximum number of results.
            similarity_threshold: Minimum similarity score (0-1).
            
        Returns:
            List of matching embeddings with similarity scores.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # Build search query
            cursor = self.db.cursor()
            
            if content_type:
                search_sql = """
                SELECT 
                    id,
                    project_id,
                    content_type,
                    content_id,
                    metadata,
                    created_at,
                    1 - (embedding <=> %s::vector) as similarity
                FROM project_embeddings
                WHERE project_id = %s AND content_type = %s
                AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """
                
                cursor.execute(search_sql, (
                    embedding_str,
                    project_id,
                    content_type,
                    embedding_str,
                    similarity_threshold,
                    embedding_str,
                    limit
                ))
            else:
                search_sql = """
                SELECT 
                    id,
                    project_id,
                    content_type,
                    content_id,
                    metadata,
                    created_at,
                    1 - (embedding <=> %s::vector) as similarity
                FROM project_embeddings
                WHERE project_id = %s
                AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """
                
                cursor.execute(search_sql, (
                    embedding_str,
                    project_id,
                    embedding_str,
                    similarity_threshold,
                    embedding_str,
                    limit
                ))
            
            results = cursor.fetchall()
            cursor.close()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row[0],
                    "project_id": row[1],
                    "content_type": row[2],
                    "content_id": row[3],
                    "metadata": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                    "similarity": float(row[6])
                })
            
            logger.debug(f"Semantic search found {len(formatted_results)} results")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    def delete_embedding(self, embedding_id: int) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            embedding_id: ID of embedding to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        try:
            cursor = self.db.cursor()
            
            delete_sql = "DELETE FROM project_embeddings WHERE id = %s;"
            cursor.execute(delete_sql, (embedding_id,))
            
            deleted = cursor.rowcount > 0
            self.db.commit()
            
            if deleted:
                logger.info(f"Deleted embedding {embedding_id}")
            
            return deleted
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete embedding: {e}")
            raise
    
    def delete_project_embeddings(self, project_id: str) -> int:
        """
        Delete all embeddings for a project.
        
        Args:
            project_id: Project identifier.
            
        Returns:
            Number of embeddings deleted.
        """
        try:
            cursor = self.db.cursor()
            
            delete_sql = "DELETE FROM project_embeddings WHERE project_id = %s;"
            cursor.execute(delete_sql, (project_id,))
            
            deleted_count = cursor.rowcount
            self.db.commit()
            
            logger.info(f"Deleted {deleted_count} embeddings for project {project_id}")
            return deleted_count
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete project embeddings: {e}")
            raise
