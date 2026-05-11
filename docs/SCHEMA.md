# Esquema de Banco de Dados

O CARQ utiliza PostgreSQL com pgvector.

## Tabelas

### `rag_documents`
Registro raiz do documento.

Campos:
- `id` UUID PK
- `source_uri` varchar(2048)
- `content_hash` bytea
- `status` varchar(32)
- `document_type` varchar(50)
- `attributes` jsonb
- `error_message` text
- `created_at` timestamptz
- `updated_at` timestamptz
- `completed_at` timestamptz

Índices:
- `idx_doc_status_created(status, created_at)`
- `idx_doc_uri_hash(source_uri, content_hash)` unique

### `rag_chunks`
Segmentos de documento.

Campos:
- `id` UUID PK
- `document_id` UUID FK -> `rag_documents.id`
- `chunk_index` integer
- `content` text
- `content_hash` bytea
- `status` varchar(32)
- `tokens` integer
- `attributes` jsonb
- `created_at` timestamptz

Restrições:
- unique `(document_id, chunk_index)`

Índices:
- `idx_chunk_status(status)`
- `idx_chunk_hash(content_hash)`

### `rag_embeddings`
Armazenamento de vetores.

Campos:
- `id` UUID PK
- `chunk_id` UUID FK -> `rag_chunks.id`
- `text` text
- `embedding` vector(3072)
- `model` varchar(100)
- `tokens_used` integer
- `cost_usd` float
- `attributes` jsonb
- `created_at` timestamptz

Restrições:
- unique `chunk_id`

Índices:
- `idx_embedding_vector` HNSW índice cosseno em `embedding::halfvec(3072)`

### `processing_tasks`
Tabela de fila para trabalho em segundo plano.

Campos:
- `id` UUID PK
- `document_id` UUID FK -> `rag_documents.id`
- `task_type` varchar(50)
- `status` varchar(32)
- `priority` integer
- `attempt_count` integer
- `max_attempts` integer
- `error_message` text
- `attributes` jsonb
- `created_at` timestamptz
- `updated_at` timestamptz
- `completed_at` timestamptz
- `worker_id` varchar(255)

Índices:
- `idx_task_status_priority(status, priority)`
- `idx_task_created(created_at)`
- `idx_task_worker(worker_id)`

### `task_deadletter`
Tarefas que falharam e excederam os limites de tentativas.

Campos:
- `id` UUID PK
- `task_id` UUID
- `document_id` UUID FK -> `rag_documents.id`
- `reason` text
- `error_details` jsonb
- `created_at` timestamptz

Índices:
- `idx_dlq_task_id(task_id)`
- `idx_dlq_created(created_at)`

## Enumerações

### `DocumentStatus`
- `pending`
- `chunking`
- `embedding`
- `done`
- `failed`
- `archived`

### `ChunkStatus`
- `pending`
- `embedding`
- `done`
- `failed`

### `TaskType`
- `parse_pdf`
- `chunk_document`
- `embed_chunk`
- `insert_vector`

### `TaskStatus`
- `pending`
- `processing`
- `done`
- `failed`
- `retrying`

## Relacionamentos

- `Document` possui muitos `Chunk`
- `Document` possui muitos `ProcessingTask`
- `Document` possui muitos `TaskDeadletter`
- `Chunk` possui um `Embedding`

