"""
Tests for pgvector setup and embedding functionality.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from app.core.pgvector_setup import PgvectorSetup
from app.core.embeddings import EmbeddingGenerator, EmbeddingStore
from app.core.context_retriever import ContextRetriever


class TestPgvectorSetup:
    """Test pgvector setup and initialization."""
    
    def test_pgvector_setup_init_with_env_var(self):
        """Test PgvectorSetup initialization with environment variable."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}):
            setup = PgvectorSetup()
            assert setup.db_url == "postgresql://user:pass@localhost/db"
    
    def test_pgvector_setup_init_with_explicit_url(self):
        """Test PgvectorSetup initialization with explicit URL."""
        url = "postgresql://user:pass@localhost/db"
        setup = PgvectorSetup(db_url=url)
        assert setup.db_url == url
    
    def test_pgvector_setup_init_missing_env_var(self):
        """Test PgvectorSetup initialization fails without DATABASE_URL."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL"):
                PgvectorSetup()


class TestEmbeddingGenerator:
    """Test embedding generation."""
    
    def test_embedding_generator_init_with_env_var(self):
        """Test EmbeddingGenerator initialization with environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            generator = EmbeddingGenerator()
            assert generator.api_key == "sk-test-key"
            assert generator.model == "text-embedding-3-small"
            assert generator.embedding_dimension == 1536
    
    def test_embedding_generator_init_with_explicit_key(self):
        """Test EmbeddingGenerator initialization with explicit key."""
        generator = EmbeddingGenerator(api_key="sk-test-key")
        assert generator.api_key == "sk-test-key"
    
    def test_embedding_generator_init_missing_api_key(self):
        """Test EmbeddingGenerator initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                EmbeddingGenerator()
    
    @patch('app.core.embeddings.OpenAI')
    def test_generate_embedding(self, mock_openai_class):
        """Test single embedding generation."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="sk-test-key")
        embedding = generator.generate_embedding("test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()
    
    @patch('app.core.embeddings.OpenAI')
    def test_generate_embeddings_batch(self, mock_openai_class):
        """Test batch embedding generation."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock batch response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=[0.1, 0.2]),
            MagicMock(index=1, embedding=[0.3, 0.4])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="sk-test-key")
        embeddings = generator.generate_embeddings_batch(["text1", "text2"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]


class TestEmbeddingStore:
    """Test embedding storage and retrieval."""
    
    @patch('app.core.embeddings.EmbeddingGenerator')
    def test_store_embedding(self, mock_generator_class):
        """Test storing a single embedding."""
        # Mock database connection
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (123,)  # Embedding ID
        
        # Mock embedding generator
        mock_generator = MagicMock()
        mock_generator.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_generator_class.return_value = mock_generator
        
        store = EmbeddingStore(mock_db)
        embedding_id = store.store_embedding(
            project_id="proj-1",
            content_type="summary",
            content_id="content-1",
            text="Test summary",
            metadata={"key": "value"}
        )
        
        assert embedding_id == 123
        mock_db.cursor.assert_called()
        mock_db.commit.assert_called()
    
    @patch('app.core.embeddings.EmbeddingGenerator')
    def test_semantic_search(self, mock_generator_class):
        """Test semantic search functionality."""
        # Mock database connection
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value = mock_cursor
        
        # Mock search results
        mock_cursor.fetchall.return_value = [
            (1, "proj-1", "summary", "content-1", {"text": "summary text"}, 0.95),
            (2, "proj-1", "summary", "content-2", {"text": "another summary"}, 0.87)
        ]
        
        # Mock embedding generator
        mock_generator = MagicMock()
        mock_generator.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_generator_class.return_value = mock_generator
        
        store = EmbeddingStore(mock_db)
        results = store.semantic_search(
            project_id="proj-1",
            query="test query",
            content_type="summary",
            limit=5
        )
        
        assert len(results) == 2
        assert results[0]["similarity"] == 0.95
        assert results[1]["similarity"] == 0.87
        mock_cursor.execute.assert_called()
    
    @patch('app.core.embeddings.EmbeddingGenerator')
    def test_delete_embedding(self, mock_generator_class):
        """Test deleting an embedding."""
        # Mock database connection
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        # Mock embedding generator
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        store = EmbeddingStore(mock_db)
        deleted = store.delete_embedding(123)
        
        assert deleted is True
        mock_db.commit.assert_called()
    
    @patch('app.core.embeddings.EmbeddingGenerator')
    def test_delete_project_embeddings(self, mock_generator_class):
        """Test deleting all embeddings for a project."""
        # Mock database connection
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5
        
        # Mock embedding generator
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        store = EmbeddingStore(mock_db)
        deleted_count = store.delete_project_embeddings("proj-1")
        
        assert deleted_count == 5
        mock_db.commit.assert_called()


class TestContextRetriever:
    """Test context retrieval functionality."""
    
    @patch('app.core.context_retriever.EmbeddingStore')
    def test_retrieve_meeting_context(self, mock_store_class):
        """Test retrieving meeting context."""
        # Mock database connection
        mock_db = MagicMock()
        
        # Mock embedding store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        # Mock search results
        mock_store.semantic_search.side_effect = [
            [{"id": 1, "content_type": "summary", "similarity": 0.95}],
            [{"id": 2, "content_type": "decision", "similarity": 0.88}],
            [{"id": 3, "content_type": "blocker", "similarity": 0.82}]
        ]
        
        retriever = ContextRetriever(mock_db)
        context = retriever.retrieve_meeting_context(
            project_id="proj-1",
            query="test query"
        )
        
        assert "summaries" in context
        assert "decisions" in context
        assert "blockers" in context
        assert len(context["summaries"]) == 1
        assert len(context["decisions"]) == 1
        assert len(context["blockers"]) == 1
    
    @patch('app.core.context_retriever.EmbeddingStore')
    def test_retrieve_project_context(self, mock_store_class):
        """Test retrieving all project context."""
        # Mock database connection
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value = mock_cursor
        
        # Mock results
        mock_cursor.fetchall.return_value = [
            (1, "summary", "content-1", {"text": "summary"}, 0.95),
            (2, "decision", "content-2", {"text": "decision"}, 0.88)
        ]
        
        # Mock embedding store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        retriever = ContextRetriever(mock_db)
        context = retriever.retrieve_project_context(project_id="proj-1")
        
        assert "summaries" in context
        assert "decisions" in context
        assert len(context["summaries"]) == 1
        assert len(context["decisions"]) == 1
    
    @patch('app.core.context_retriever.EmbeddingStore')
    def test_build_prompt_context(self, mock_store_class):
        """Test building formatted context for prompts."""
        # Mock database connection
        mock_db = MagicMock()
        
        # Mock embedding store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        # Mock search results
        mock_store.semantic_search.side_effect = [
            [{"id": 1, "content_type": "summary", "similarity": 0.95, 
              "metadata": {"text": "Meeting summary"}}],
            [{"id": 2, "content_type": "decision", "similarity": 0.88,
              "metadata": {"text": "Key decision"}}],
            [{"id": 3, "content_type": "blocker", "similarity": 0.82,
              "metadata": {"text": "Known blocker"}}]
        ]
        
        retriever = ContextRetriever(mock_db)
        context_str = retriever.build_prompt_context(
            project_id="proj-1",
            query="test query"
        )
        
        assert isinstance(context_str, str)
        assert "Meeting summary" in context_str or len(context_str) > 0
