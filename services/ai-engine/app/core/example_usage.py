"""
Example usage of pgvector setup and semantic search.
This file demonstrates how to use the embedding and context retrieval modules.
"""

import os
import psycopg2
from app.core.pgvector_setup import PgvectorSetup
from app.core.embeddings import EmbeddingStore
from app.core.context_retriever import ContextRetriever


def example_setup():
    """Example: Initialize pgvector in PostgreSQL."""
    print("=== Example 1: Initialize pgvector ===")
    
    setup = PgvectorSetup()
    setup.initialize()
    print("✓ pgvector initialized successfully")


def example_store_embeddings():
    """Example: Store embeddings for meeting content."""
    print("\n=== Example 2: Store Embeddings ===")
    
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    # Create embedding store
    store = EmbeddingStore(conn)
    
    # Store a meeting summary
    summary_id = store.store_embedding(
        project_id="proj-acme",
        content_type="summary",
        content_id="meeting-2024-01-15",
        text="In today's meeting, we discussed the Q1 roadmap. Key decisions: "
             "prioritize mobile app development, delay backend refactoring to Q2, "
             "allocate 2 engineers to performance optimization.",
        metadata={
            "meeting_date": "2024-01-15",
            "attendees": ["Alice", "Bob", "Charlie"],
            "duration_minutes": 45
        }
    )
    print(f"✓ Stored summary embedding: {summary_id}")
    
    # Store a decision
    decision_id = store.store_embedding(
        project_id="proj-acme",
        content_type="decision",
        content_id="decision-mobile-priority",
        text="Mobile app development is now the top priority for Q1",
        metadata={"decided_by": "Alice", "date": "2024-01-15"}
    )
    print(f"✓ Stored decision embedding: {decision_id}")
    
    # Store a blocker
    blocker_id = store.store_embedding(
        project_id="proj-acme",
        content_type="blocker",
        content_id="blocker-api-latency",
        text="API response times are exceeding SLA due to database queries",
        metadata={"severity": "high", "reported_by": "Bob"}
    )
    print(f"✓ Stored blocker embedding: {blocker_id}")
    
    conn.close()


def example_semantic_search():
    """Example: Perform semantic search on embeddings."""
    print("\n=== Example 3: Semantic Search ===")
    
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    # Create embedding store
    store = EmbeddingStore(conn)
    
    # Search for relevant content
    query = "What are the main priorities for this quarter?"
    results = store.semantic_search(
        project_id="proj-acme",
        query=query,
        limit=5,
        similarity_threshold=0.5
    )
    
    print(f"Search query: '{query}'")
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  - {result['content_type']}: {result['content_id']} "
              f"(similarity: {result['similarity']:.2f})")
    
    conn.close()


def example_context_retrieval():
    """Example: Retrieve context for AI agents."""
    print("\n=== Example 4: Context Retrieval ===")
    
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    # Create context retriever
    retriever = ContextRetriever(conn)
    
    # Get meeting context for a query
    query = "What are the current blockers and risks?"
    context = retriever.retrieve_meeting_context(
        project_id="proj-acme",
        query=query,
        limit=3
    )
    
    print(f"Context for query: '{query}'")
    print(f"  Summaries: {len(context['summaries'])} found")
    print(f"  Decisions: {len(context['decisions'])} found")
    print(f"  Blockers: {len(context['blockers'])} found")
    
    # Build formatted context for LLM prompt
    prompt_context = retriever.build_prompt_context(
        project_id="proj-acme",
        query=query,
        max_tokens=2000
    )
    
    print(f"\nFormatted context for LLM ({len(prompt_context)} chars):")
    print(prompt_context[:500] + "..." if len(prompt_context) > 500 else prompt_context)
    
    conn.close()


def example_batch_operations():
    """Example: Store multiple embeddings in batch."""
    print("\n=== Example 5: Batch Operations ===")
    
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    # Create embedding store
    store = EmbeddingStore(conn)
    
    # Prepare batch items
    items = [
        {
            "content_id": "meeting-2024-01-16",
            "text": "Discussed API performance issues and optimization strategies",
            "metadata": {"meeting_date": "2024-01-16"}
        },
        {
            "content_id": "meeting-2024-01-17",
            "text": "Reviewed Q1 progress and adjusted timeline for mobile app",
            "metadata": {"meeting_date": "2024-01-17"}
        },
        {
            "content_id": "meeting-2024-01-18",
            "text": "Planning session for backend refactoring in Q2",
            "metadata": {"meeting_date": "2024-01-18"}
        }
    ]
    
    # Store in batch
    embedding_ids = store.store_embeddings_batch(
        project_id="proj-acme",
        content_type="summary",
        items=items
    )
    
    print(f"✓ Stored {len(embedding_ids)} embeddings in batch")
    for i, embedding_id in enumerate(embedding_ids):
        print(f"  - {items[i]['content_id']}: {embedding_id}")
    
    conn.close()


def example_similar_content():
    """Example: Find similar content."""
    print("\n=== Example 6: Find Similar Content ===")
    
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    # Create context retriever
    retriever = ContextRetriever(conn)
    
    # Find content similar to a specific item
    similar = retriever.retrieve_similar_content(
        project_id="proj-acme",
        content_id="meeting-2024-01-15",
        limit=3
    )
    
    print(f"Content similar to 'meeting-2024-01-15':")
    for item in similar:
        print(f"  - {item['content_type']}: {item['content_id']} "
              f"(similarity: {item['similarity']:.2f})")
    
    conn.close()


def example_recent_context():
    """Example: Get recent context."""
    print("\n=== Example 7: Recent Context ===")
    
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    # Create context retriever
    retriever = ContextRetriever(conn)
    
    # Get context from last 7 days
    context = retriever.get_recent_context(
        project_id="proj-acme",
        days=7,
        limit=5
    )
    
    print(f"Recent context ({context['time_range']}):")
    print(f"  Summaries: {len(context['summaries'])}")
    print(f"  Decisions: {len(context['decisions'])}")
    print(f"  Blockers: {len(context['blockers'])}")
    print(f"  Other: {len(context['other'])}")
    
    conn.close()


if __name__ == "__main__":
    """Run all examples."""
    try:
        # Uncomment to run examples
        # example_setup()
        # example_store_embeddings()
        # example_semantic_search()
        # example_context_retrieval()
        # example_batch_operations()
        # example_similar_content()
        # example_recent_context()
        
        print("Examples are available but commented out.")
        print("Uncomment the example functions to run them.")
        print("\nNote: Requires DATABASE_URL and OPENAI_API_KEY environment variables")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
