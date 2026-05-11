"""Create initial schema with rag_documents and processing_tasks tables.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-05-10 12:30:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create extension for UUID support
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # rag_documents table
    op.create_table(
        "rag_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("source_uri", sa.String(2048), nullable=False, index=True),
        sa.Column("content_hash", sa.LargeBinary(), nullable=False, index=True),
        sa.Column("status", sa.String(32), nullable=False, default="pending", index=True),
        sa.Column("document_type", sa.String(50), nullable=False, default="pdf"),
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Index("idx_doc_status_created", "status", "created_at"),
        sa.Index("idx_doc_uri_hash", "source_uri", "content_hash", unique=True),
    )

    # rag_chunks table
    op.create_table(
        "rag_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.LargeBinary(), nullable=False, index=True),
        sa.Column("status", sa.String(32), nullable=False, default="pending", index=True),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_index"),
        sa.Index("idx_chunk_status", "status"),
        sa.Index("idx_chunk_hash", "content_hash"),
    )

    # rag_embeddings table (with pgvector support)
    op.create_table(
        "rag_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True, index=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),  # Stored via pgvector VECTOR(3072)
        sa.Column("model", sa.String(100), nullable=False, default="text-embedding-3-large"),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["chunk_id"], ["rag_chunks.id"], ondelete="CASCADE"),
    )

    # Add vector column with proper pgvector type
    op.execute("ALTER TABLE rag_embeddings DROP COLUMN embedding;")
    op.execute("ALTER TABLE rag_embeddings ADD COLUMN embedding vector(3072) NOT NULL;")

    # processing_tasks table (queue)
    op.create_table(
        "processing_tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("task_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, default="pending", index=True),
        sa.Column("priority", sa.Integer(), nullable=False, default=0, index=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, default=0),
        sa.Column("max_attempts", sa.Integer(), nullable=False, default=3),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], ondelete="CASCADE"),
        sa.Index("idx_task_status_priority", "status", "priority"),
        sa.Index("idx_task_created", "created_at"),
        sa.Index("idx_task_worker", "worker_id"),
    )

    # task_deadletter table
    op.create_table(
        "task_deadletter",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("error_details", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], ondelete="CASCADE"),
        sa.Index("idx_dlq_task_id", "task_id"),
        sa.Index("idx_dlq_created", "created_at"),
    )

    # Create HNSW index for vector similarity search (supports >2000 dims via halfvec cast)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding_vector ON rag_embeddings
        USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)


def downgrade() -> None:
    op.drop_table("task_deadletter")
    op.drop_table("processing_tasks")
    op.drop_table("rag_embeddings")
    op.drop_table("rag_chunks")
    op.drop_table("rag_documents")
