# Referência da API

Caminho base: `/api/v1`

A autenticação utiliza o cabeçalho `X-API-Key`. Requisições sem uma chave configurada retornam `401` ou `503` se o repositório de chaves não estiver configurado.

## Endpoints

### `POST /embed`
Gera um único embedding.

Requisição:
```json
{
  "text": "Hello world",
  "model": "text-embedding-3-small",
  "metadata": {}
}
```

Validação:
- `text`: 1..10000 caracteres
- `model`: `text-embedding-3-large | text-embedding-3-small | text-embedding-ada-002`

### `POST /embed-batch`
Gera embeddings para até 100 textos.

Requisição:
```json
{
  "texts": ["First document", "Second document"],
  "model": "text-embedding-3-small"
}
```

Validação:
- `texts`: 1..100 itens
- cada texto: mesmos limites de `/embed`

### `POST /search`
Executa busca semântica nos vetores armazenados.

Requisição:
```json
{
  "query": "machine learning models",
  "model": "text-embedding-3-small",
  "limit": 10,
  "similarity_threshold": 0.7
}
```

Validação:
- `limit`: 1..100
- `similarity_threshold`: 0.0..1.0

### `GET /stats`
Retorna métricas de uso e armazenamento.

### `GET /health`
Verificação de saúde leve da API.

### `POST /documents`
Aceita um documento para ingestão assíncrona.

Requisição:
```json
{
  "source_uri": "https://example.com/document.pdf",
  "document_type": "pdf",
  "priority": 0,
  "attributes": {}
}
```

Valores de `document_type`:
- `pdf`
- `text`
- `url`

### `GET /documents/{document_id}`
Retorna o status de processamento de um documento.

### `GET /documents`
Lista documentos com paginação.

Parâmetros de consulta:
- `limit` (padrão 20, máximo 100)
- `offset` (padrão 0)
- `status` (filtro opcional)

### `POST /documents/{document_id}/retry`
Recoloca as tarefas de processamento do documento na fila.

## Formatos de resposta

### `EmbedResponse`
- `text`
- `embedding`
- `model`
- `tokens_used`
- `cost_usd`
- `cached`

### `SearchResponse`
- `query`
- `results[]`
- `count`
- `search_latency_ms`

### `IngestResponse`
- `document_id`
- `status`
- `message`
- `task_id`

### `DocumentStatusResponse`
- `document_id`
- `status`
- `document_type`
- `source_uri`
- `chunk_count`
- `embedding_count`
- `progress_percentage`
- `error_message`

## Respostas de erro

Todos os erros da API utilizam:
```json
{
  "error": "Invalid API key",
  "detail": "Authentication failed",
  "status_code": 401
}
```

