"""
Database migration utilities for pgvector setup.
Provides functions to initialize and manage pgvector schema.
"""

import logging
import os
from app.core.pgvector_setup import PgvectorSetup

logger = logging.getLogger(__name__)


def migrate_pgvector():
    """
    Run pgvector migration to set up embeddings infrastructure.
    
    This function:
    1. Creates pgvector extension
    2. Creates project_embeddings table
    3. Creates indexes for efficient search
    
    Should be called during application startup or as a management command.
    """
    try:
        logger.info("Starting pgvector migration...")
        setup = PgvectorSetup()
        setup.initialize()
        logger.info("pgvector migration completed successfully")
        return True
    except Exception as e:
        logger.error(f"pgvector migration failed: {e}")
        raise


def rollback_pgvector():
    """
    Rollback pgvector migration (drops tables and extension).
    
    WARNING: This will delete all embeddings data!
    """
    try:
        logger.warning("Starting pgvector rollback - this will delete all embeddings!")
        setup = PgvectorSetup()
        conn = setup.get_connection()
        cursor = conn.cursor()
        
        # Drop indexes
        cursor.execute("DROP INDEX IF EXISTS idx_project_embeddings_embedding;")
        cursor.execute("DROP INDEX IF EXISTS idx_project_embeddings_project_id;")
        cursor.execute("DROP INDEX IF EXISTS idx_project_embeddings_content_type;")
        
        # Drop table
        cursor.execute("DROP TABLE IF EXISTS project_embeddings;")
        
        # Drop extension
        cursor.execute("DROP EXTENSION IF EXISTS vector;")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("pgvector rollback completed")
        return True
    except Exception as e:
        logger.error(f"pgvector rollback failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_pgvector()
