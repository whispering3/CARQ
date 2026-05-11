-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector index type if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type WHERE typname = 'vector'
    ) THEN
        CREATE TYPE vector AS (
            dim int4,
            data float8[]
        );
    END IF;
END
$$;
