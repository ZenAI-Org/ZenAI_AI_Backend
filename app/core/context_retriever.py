"""
Context retrieval module for semantic search and memory management.
Handles retrieval of relevant context for AI agents with caching optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from app.core.embeddings import EmbeddingStore
from app.core.performance_optimizer import (
    get_cache_manager,
    get_performance_metrics,
    CacheManager,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieves relevant context for AI agents using semantic search with caching."""
    
    def __init__(
        self,
        db_connection,
        cache_manager: Optional[CacheManager] = None,
        metrics: Optional[PerformanceMetrics] = None
    ):
        """
        Initialize context retriever.
        
        Args:
            db_connection: PostgreSQL connection object.
            cache_manager: Optional cache manager for caching results
            metrics: Optional performance metrics tracker
        """
        self.db = db_connection
        self.embedding_store = EmbeddingStore(db_connection)
        self.cache_manager = cache_manager or get_cache_manager()
        self.metrics = metrics or get_performance_metrics()
    
    def retrieve_meeting_context(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant meeting context for a query with caching.
        
        Args:
            project_id: Project identifier.
            query: Query text.
            limit: Maximum number of results per content type.
            use_cache: Whether to use cache for results.
            
        Returns:
            Dictionary with summaries, decisions, and blockers.
        """
        try:
            # Try to get from cache
            if use_cache:
                cached_context = self.cache_manager.get_context(
                    project_id=project_id,
                    query=query,
                    limit=limit
                )
                if cached_context:
                    self.metrics.record_cache_hit("context")
                    logger.debug(f"Cache hit for context: {project_id}:{query}")
                    return cached_context
                
                self.metrics.record_cache_miss("context")
            
            context = {
                "summaries": [],
                "decisions": [],
                "blockers": [],
                "query": query
            }
            
            # Search for summaries
            summaries = self.embedding_store.semantic_search(
                project_id=project_id,
                query=query,
                content_type="summary",
                limit=limit,
                similarity_threshold=0.5
            )
            context["summaries"] = summaries
            
            # Search for decisions
            decisions = self.embedding_store.semantic_search(
                project_id=project_id,
                query=query,
                content_type="decision",
                limit=limit,
                similarity_threshold=0.5
            )
            context["decisions"] = decisions
            
            # Search for blockers
            blockers = self.embedding_store.semantic_search(
                project_id=project_id,
                query=query,
                content_type="blocker",
                limit=limit,
                similarity_threshold=0.5
            )
            context["blockers"] = blockers
            
            # Cache the result
            if use_cache:
                self.cache_manager.set_context(
                    project_id=project_id,
                    query=query,
                    context=context,
                    limit=limit
                )
            
            logger.info(f"Retrieved context for project {project_id}: "
                       f"{len(summaries)} summaries, {len(decisions)} decisions, "
                       f"{len(blockers)} blockers")
            
            return context
        
        except Exception as e:
            logger.error(f"Failed to retrieve meeting context: {e}")
            raise
    
    def retrieve_project_context(
        self,
        project_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve all relevant context for a project.
        
        Args:
            project_id: Project identifier.
            limit: Maximum number of results per content type.
            
        Returns:
            Dictionary with all project context.
        """
        try:
            cursor = self.db.cursor()
            
            # Get all embeddings for the project
            query_sql = """
            SELECT 
                id,
                content_type,
                content_id,
                metadata,
                1 - (embedding <=> (
                    SELECT AVG(embedding) FROM project_embeddings 
                    WHERE project_id = %s
                )) as similarity
            FROM project_embeddings
            WHERE project_id = %s
            ORDER BY created_at DESC
            LIMIT %s;
            """
            
            cursor.execute(query_sql, (project_id, project_id, limit))
            results = cursor.fetchall()
            cursor.close()
            
            # Organize by content type
            context = {
                "summaries": [],
                "decisions": [],
                "blockers": [],
                "other": []
            }
            
            for row in results:
                item = {
                    "id": row[0],
                    "content_type": row[1],
                    "content_id": row[2],
                    "metadata": row[3],
                    "similarity": float(row[4]) if row[4] else 0
                }
                
                if row[1] == "summary":
                    context["summaries"].append(item)
                elif row[1] == "decision":
                    context["decisions"].append(item)
                elif row[1] == "blocker":
                    context["blockers"].append(item)
                else:
                    context["other"].append(item)
            
            logger.info(f"Retrieved project context for {project_id}")
            return context
        
        except Exception as e:
            logger.error(f"Failed to retrieve project context: {e}")
            raise
    
    def retrieve_similar_content(
        self,
        project_id: str,
        content_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve content similar to a given content item.
        
        Args:
            project_id: Project identifier.
            content_id: Content identifier to find similar items for.
            limit: Maximum number of results.
            
        Returns:
            List of similar content items.
        """
        try:
            # Try to get from cache
            cache_key = self.cache_manager._make_key("similar", project_id, content_id, limit)
            cached_results = self.cache_manager.get(cache_key)
            if cached_results:
                self.metrics.record_cache_hit("similar_content")
                return cached_results
            
            self.metrics.record_cache_miss("similar_content")

            cursor = self.db.cursor()
            
            # Get the embedding for the content
            get_embedding_sql = """
            SELECT embedding FROM project_embeddings 
            WHERE project_id = %s AND content_id = %s
            LIMIT 1;
            """
            
            cursor.execute(get_embedding_sql, (project_id, content_id))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Content {content_id} not found in project {project_id}")
                return []
            
            # Find similar embeddings
            search_sql = """
            SELECT 
                id,
                content_type,
                content_id,
                metadata,
                1 - (embedding <=> %s) as similarity
            FROM project_embeddings
            WHERE project_id = %s AND content_id != %s
            AND 1 - (embedding <=> %s) > 0.5
            ORDER BY embedding <=> %s
            LIMIT %s;
            """
            
            cursor.execute(search_sql, (
                result[0],
                project_id,
                content_id,
                result[0],
                result[0],
                limit
            ))
            
            results = cursor.fetchall()
            cursor.close()
            
            # Format results
            similar_items = []
            for row in results:
                similar_items.append({
                    "id": row[0],
                    "content_type": row[1],
                    "content_id": row[2],
                    "metadata": row[3],
                    "similarity": float(row[4])
                })
            
            logger.debug(f"Found {len(similar_items)} similar items to {content_id}")
            
            # Cache the results
            self.cache_manager.set(cache_key, similar_items)
            
            return similar_items
        
        except Exception as e:
            logger.error(f"Failed to retrieve similar content: {e}")
            raise
    
    def get_recent_context(
        self,
        project_id: str,
        days: int = 7,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve recent context from the past N days.
        
        Args:
            project_id: Project identifier.
            days: Number of days to look back.
            limit: Maximum number of results per content type.
            
        Returns:
            Dictionary with recent context organized by type.
        """
        try:
            cursor = self.db.cursor()
            
            # Get recent embeddings
            query_sql = """
            SELECT 
                id,
                content_type,
                content_id,
                metadata,
                created_at
            FROM project_embeddings
            WHERE project_id = %s 
            AND created_at >= NOW() - INTERVAL '%s days'
            ORDER BY created_at DESC
            LIMIT %s;
            """
            
            cursor.execute(query_sql, (project_id, days, limit))
            results = cursor.fetchall()
            cursor.close()
            
            # Organize by content type
            context = {
                "summaries": [],
                "decisions": [],
                "blockers": [],
                "other": [],
                "time_range": f"Last {days} days"
            }
            
            for row in results:
                item = {
                    "id": row[0],
                    "content_type": row[1],
                    "content_id": row[2],
                    "metadata": row[3],
                    "created_at": row[4].isoformat() if row[4] else None
                }
                
                if row[1] == "summary":
                    context["summaries"].append(item)
                elif row[1] == "decision":
                    context["decisions"].append(item)
                elif row[1] == "blocker":
                    context["blockers"].append(item)
                else:
                    context["other"].append(item)
            
            logger.info(f"Retrieved recent context for {project_id} from last {days} days")
            return context
        
        except Exception as e:
            logger.error(f"Failed to retrieve recent context: {e}")
            raise
    
    def build_prompt_context(
        self,
        project_id: str,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Build a formatted context string for LLM prompts.
        
        Args:
            project_id: Project identifier.
            query: Query text for semantic search.
            max_tokens: Maximum tokens to include (approximate).
            
        Returns:
            Formatted context string for use in prompts.
        """
        try:
            context_data = self.retrieve_meeting_context(
                project_id=project_id,
                query=query,
                limit=5
            )
            
            context_parts = []
            
            # Add summaries
            if context_data["summaries"]:
                context_parts.append("## Recent Meeting Summaries")
                for summary in context_data["summaries"]:
                    metadata = summary.get("metadata", {})
                    text = metadata.get("text", "")
                    if text:
                        context_parts.append(f"- {text} (similarity: {summary['similarity']:.2f})")
            
            # Add decisions
            if context_data["decisions"]:
                context_parts.append("\n## Key Decisions")
                for decision in context_data["decisions"]:
                    metadata = decision.get("metadata", {})
                    text = metadata.get("text", "")
                    if text:
                        context_parts.append(f"- {text} (similarity: {decision['similarity']:.2f})")
            
            # Add blockers
            if context_data["blockers"]:
                context_parts.append("\n## Known Blockers")
                for blocker in context_data["blockers"]:
                    metadata = blocker.get("metadata", {})
                    text = metadata.get("text", "")
                    if text:
                        context_parts.append(f"- {text} (similarity: {blocker['similarity']:.2f})")
            
            context_str = "\n".join(context_parts)
            
            # Truncate if needed
            if len(context_str) > max_tokens * 4:  # Rough estimate: 1 token ≈ 4 chars
                context_str = context_str[:max_tokens * 4]
            
            logger.debug(f"Built prompt context with {len(context_str)} characters")
            return context_str
        
        except Exception as e:
            logger.error(f"Failed to build prompt context: {e}")
            raise
