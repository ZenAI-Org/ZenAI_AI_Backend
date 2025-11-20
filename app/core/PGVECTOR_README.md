# pgvector Setup and Semantic Search

This module provides pgvector integration for semantic search and embeddings storage in PostgreSQL.

## Overview

The pgvector setup enables:
- **Semantic Search**: Find relevant context using vector similarity
- **Embeddings Storage**: Store OpenAI embeddings in PostgreSQL
- **Context Retrieval**: Retrieve relevant meeting summaries, decisions, and blockers for AI agents
- **Memory Management**: Maintain project context across multiple AI workflows

## Components

### 1. PgvectorSetup (`pgvector_setup.py`)

Handles PostgreSQL pgvector extension and table initialization.

**Key Methods:**
- `create_pgvector_extension()`: Creates pgvector extension in PostgreSQL
- `create_project_embeddings_table()`: Creates the embeddings table
- `create_indexes()`: Creates IVFFlat indexes for efficient similarity search
- `initialize()`: Runs all initialization steps

**Usage:**
```python
from app.core.pgvector_setup import PgvectorSetup

setup = PgvectorSetup()
setup.initialize()
```

### 2. EmbeddingGenerator (`embeddings.py`)

Generates embeddings using OpenAI API.

**Key Methods:**
- `generate_embedding(text)`: Generate embedding for single text
- `generate_embeddings_batch(texts)`: Generate embeddings for multiple texts

**Usage:**
```python
from app.core.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()
embedding = generator.generate_embedding("Meeting summary text")
```

### 3. EmbeddingStore (`embeddings.py`)

Stores and retrieves embeddings from PostgreSQL.

**Key Methods:**
- `store_embedding()`: Store single embedding with metadata
- `store_embeddings_batch()`: Store multiple embeddings
- `semantic_search()`: Search for similar embeddings
- `delete_embedding()`: Delete specific embedding
- `delete_project_embeddings()`: Delete all embeddings for a project

**Usage:**
```python
from app.core.embeddings import EmbeddingStore
import psycopg2

conn = psycopg2.connect("postgresql://user:pass@localhost/db")
store = EmbeddingStore(conn)

# Store embedding
embedding_id = store.store_embedding(
    project_id="proj-1",
    content_type="summary",
    content_id="meeting-1",
    text="Meeting summary text",
    metadata={"meeting_date": "2024-01-15"}
)

# Search for similar content
results = store.semantic_search(
    project_id="proj-1",
    query="What was discussed about project timeline?",
    content_type="summary",
    limit=5
)
```

### 4. ContextRetriever (`context_retriever.py`)

Retrieves relevant context for AI agents using semantic search.

**Key Methods:**
- `retrieve_meeting_context()`: Get summaries, decisions, and blockers for a query
- `retrieve_project_context()`: Get all context for a project
- `retrieve_similar_content()`: Find content similar to a specific item
- `get_recent_context()`: Get context from past N days
- `build_prompt_context()`: Format context for LLM prompts

**Usage:**
```python
from app.core.context_retriever import ContextRetriever

retriever = ContextRetriever(db_connection)

# Get context for a query
context = retriever.retrieve_meeting_context(
    project_id="proj-1",
    query="What are the current blockers?"
)

# Build formatted context for LLM
prompt_context = retriever.build_prompt_context(
    project_id="proj-1",
    query="Summarize recent decisions"
)
```

## Database Schema

### project_embeddings Table

```sql
CREATE TABLE project_embeddings (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    content_id VARCHAR(255) NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Columns:**
- `id`: Unique identifier
- `project_id`: Project identifier for filtering
- `content_type`: Type of content ('summary', 'decision', 'blocker', etc.)
- `content_id`: Unique identifier for the content
- `embedding`: 1536-dimensional vector (OpenAI embedding)
- `metadata`: JSON metadata (meeting date, source, etc.)
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Indexes

- **IVFFlat Index**: `idx_project_embeddings_embedding` - For cosine similarity search
- **Project Index**: `idx_project_embeddings_project_id` - For filtering by project
- **Content Type Index**: `idx_project_embeddings_content_type` - For filtering by type

## Setup Instructions

### 1. Prerequisites

- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- Python 3.8+

### 2. Install pgvector Extension

```bash
# On macOS with Homebrew
brew install pgvector

# Or build from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

### 3. Environment Variables

Add to `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/zenai
OPENAI_API_KEY=sk-...
```

### 4. Initialize pgvector

```python
from app.core.pgvector_setup import init_pgvector

init_pgvector()
```

Or run as a script:
```bash
python -m app.core.pgvector_setup
```

### 5. Run Migrations

```python
from app.core.migrations import migrate_pgvector

migrate_pgvector()
```

## Content Types

The system supports the following content types:

- **summary**: Meeting summaries
- **decision**: Key decisions made
- **blocker**: Known blockers or issues
- **action_item**: Action items from meetings
- **insight**: Project insights and analysis
- **recommendation**: AI recommendations

## Similarity Search

The system uses cosine similarity for semantic search. Results are ranked by similarity score (0-1).

**Example:**
```python
results = store.semantic_search(
    project_id="proj-1",
    query="What are the main risks?",
    content_type="blocker",
    limit=5,
    similarity_threshold=0.5
)

for result in results:
    print(f"{result['content_id']}: {result['similarity']:.2f}")
```

## Performance Considerations

### Indexing

The IVFFlat index is created with `lists=100`. For larger datasets, adjust this parameter:
- Smaller datasets (< 100k): `lists=10-50`
- Medium datasets (100k-1M): `lists=100-500`
- Large datasets (> 1M): `lists=500+`

### Query Optimization

1. **Filter by project_id**: Always include project_id in queries
2. **Filter by content_type**: Use content_type to narrow results
3. **Limit results**: Use reasonable limits (5-10 for most use cases)
4. **Batch operations**: Use batch methods for multiple embeddings

## Error Handling

The module includes comprehensive error handling:

```python
try:
    embedding_id = store.store_embedding(...)
except Exception as e:
    logger.error(f"Failed to store embedding: {e}")
    # Handle error appropriately
```

## Testing

Run tests with:
```bash
pytest tests/test_pgvector.py -v
```

Tests cover:
- Setup and initialization
- Embedding generation
- Storage and retrieval
- Semantic search
- Context retrieval
- Error handling

## Troubleshooting

### pgvector Extension Not Found

```
ERROR: could not open extension control file
```

**Solution**: Install pgvector extension in PostgreSQL

### Connection Refused

```
psycopg2.OperationalError: could not connect to server
```

**Solution**: Check DATABASE_URL and PostgreSQL is running

### OpenAI API Errors

```
openai.error.AuthenticationError: Invalid API key
```

**Solution**: Verify OPENAI_API_KEY is correct

### Vector Dimension Mismatch

```
ERROR: vector dimensions (1536) do not match
```

**Solution**: Ensure all embeddings use the same model (text-embedding-3-small)

## Best Practices

1. **Always include project_id**: Ensures data isolation and query performance
2. **Use metadata**: Store relevant metadata for filtering and context
3. **Batch operations**: Use batch methods for multiple embeddings
4. **Monitor similarity scores**: Adjust threshold based on use case
5. **Regular cleanup**: Delete old embeddings to maintain performance
6. **Test queries**: Verify semantic search results before production

## Future Enhancements

- [ ] Support for multiple embedding models
- [ ] Automatic embedding refresh
- [ ] Similarity threshold tuning
- [ ] Embedding quality metrics
- [ ] Batch import/export
- [ ] Embedding versioning
