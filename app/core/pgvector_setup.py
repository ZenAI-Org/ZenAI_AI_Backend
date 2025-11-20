"""
PostgreSQL pgvector setup and initialization module.
Handles creation of pgvector extension and project_embeddings table.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logger = logging.getLogger(__name__)


class PgvectorSetup:
    """Manages pgvector extension and table initialization."""
    
    def __init__(self, db_url: str = None):
        """
        Initialize pgvector setup.
        
        Args:
            db_url: PostgreSQL connection URL. If None, uses DATABASE_URL env var.
        """
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
    
    def get_connection(self):
        """Create a PostgreSQL connection."""
        try:
            conn = psycopg2.connect(self.db_url)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def create_pgvector_extension(self):
        """Create pgvector extension in PostgreSQL."""
        try:
            conn = self.get_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension created successfully")
            
            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            logger.error(f"Failed to create pgvector extension: {e}")
            raise
    
    def create_project_embeddings_table(self):
        """Create project_embeddings table with pgvector support."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create project_embeddings table
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS project_embeddings (
                id SERIAL PRIMARY KEY,
                project_id VARCHAR(255) NOT NULL,
                content_type VARCHAR(50) NOT NULL,
                content_id VARCHAR(255) NOT NULL,
                embedding vector(1536),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info("project_embeddings table created successfully")
            
            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            logger.error(f"Failed to create project_embeddings table: {e}")
            raise
    
    def create_indexes(self):
        """Create indexes for efficient similarity search."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create IVFFlat index for cosine similarity search
            create_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_embedding 
            ON project_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            
            cursor.execute(create_index_sql)
            
            # Create index on project_id for faster filtering
            create_project_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_project_id 
            ON project_embeddings (project_id);
            """
            
            cursor.execute(create_project_index_sql)
            
            # Create index on content_type for filtering
            create_content_type_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_content_type 
            ON project_embeddings (content_type);
            """
            
            cursor.execute(create_content_type_index_sql)

            # Create composite index for common queries
            create_composite_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_project_type
            ON project_embeddings (project_id, content_type);
            """
            
            cursor.execute(create_composite_index_sql)
            
            conn.commit()
            logger.info("Indexes created successfully")
            
            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    def initialize(self):
        """Run all initialization steps."""
        try:
            logger.info("Starting pgvector initialization...")
            self.create_pgvector_extension()
            self.create_project_embeddings_table()
            self.create_indexes()
            logger.info("pgvector initialization completed successfully")
        except Exception as e:
            logger.error(f"pgvector initialization failed: {e}")
            raise


def init_pgvector():
    """Convenience function to initialize pgvector."""
    setup = PgvectorSetup()
    setup.initialize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_pgvector()
