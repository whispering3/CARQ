# Arquitetura CARQ

**Context-Aware RAG Processing Queue** — Pipeline de ingestão de documentos corporativo com limitação de taxa, circuit breakers e armazenamento pgvector.

---

## Diagrama de Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              External Clients                               │
│                    (REST API consumers, PDF upload clients)                 │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ HTTPS
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Nginx Ingress Controller                            │
│                   Rate limiting · TLS termination · WAF                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ HTTP/80
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CARQ API (FastAPI)  ×3 pods                         │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Auth/JWT    │  │   /embed     │  │   /search    │  │  /health/*     │ │
│  │  Middleware  │  │   /batch     │  │   /stats     │  │  /metrics      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Rate Limiter (Token Bucket)                    │   │
│  │            3000 RPM · 1.5M TPM · Burst capacity: 1000              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────┬───────────────────────────────────┬────────────────────────┘
                 │                                   │
                 ▼                                   ▼
┌────────────────────────────┐       ┌───────────────────────────────────────┐
│   Redis (Embedding Cache)  │       │     CARQ Worker Pool  ×4 pods         │
│                            │       │                                       │
│  • TTL-based cache         │       │  ┌──────────┐  ┌──────────────────┐  │
│  • Deduplication           │       │  │ PDF      │  │  Chunk Processor │  │
│  • Session state           │       │  │ Parser   │  │  (Semantic)      │  │
└────────────────────────────┘       │  │ Workers  │  │  Workers ×8      │  │
                                     │  └────┬─────┘  └────────┬─────────┘  │
                                     │       │                  │            │
                                     │  ┌────▼──────────────────▼─────────┐ │
                                     │  │   Embedding Dispatcher           │ │
                                     │  │   (Circuit Breaker + Backoff)    │ │
                                     │  └────────────────┬─────────────────┘ │
                                     └───────────────────┼───────────────────┘
                                                         │
                            ┌────────────────────────────┼─────────────────┐
                            │                            │                 │
                            ▼                            ▼                 ▼
               ┌────────────────────┐      ┌────────────────────┐  ┌──────────────┐
               │  OpenAI Embeddings │      │  PostgreSQL         │  │  Prometheus  │
               │  API (External)    │      │  + pgvector         │  │  + Grafana   │
               │                   │      │                    │  │              │
               │  text-embedding-  │      │  • documents       │  │  • Metrics   │
               │  3-large (3072d)  │      │  • chunks          │  │  • Alerts    │
               └────────────────────┘      │  • embeddings      │  │  • Dashboards│
                                           │  (ivfflat index)   │  └──────────────┘
                                           └────────────────────┘
```

---

## Arquitetura em 4 Camadas

### Camada 1 — Camada de API (`src/carq/api/`)
A aplicação FastAPI lida com todo o tráfego HTTP de entrada. A autenticação baseada em JWT (`auth.py`) valida cada requisição. O roteador (`router.py`) despacha para os manipuladores de embedding e busca. O middleware (`monitoring/middleware.py`) injeta rastreamento de requisições, métricas do Prometheus e logging JSON estruturado em cada requisição.

### Camada 2 — Infraestrutura Central (`src/carq/core/`)
Base compartilhada utilizada por todas as outras camadas:
- **`config.py`** — Pydantic Settings com validação de variáveis de ambiente e sub-configurações tipadas (`CARQ_DB_*`, `CARQ_REDIS_*`, etc.)
- **`database.py`** — Engine assíncrona do SQLAlchemy com gerenciamento de pool de conexões
- **`exceptions.py`** — Exceções de domínio tipadas (propagadas como HTTP 4xx/5xx)
- **`logging.py`** — Logger JSON estruturado com IDs de correlação

### Camada 3 — Camada de Processamento (`src/carq/queue/`, `src/carq/pdf/`, `src/carq/embedding/`)
A lógica central do pipeline:
- **`queue/queue_manager.py`** — Fila de prioridade em memória que coordena o despacho de tarefas
- **`queue/rate_limiter.py`** — Implementação de token bucket (taxa de recarga + capacidade de burst)
- **`queue/circuit_breaker.py`** — Máquina de estados finita (FECHADO → ABERTO → SEMI-ABERTO) protegendo APIs downstream
- **`queue/backoff_strategy.py`** — Backoff exponencial + jitter para tentativas de retry
- **`pdf/pdf_parser.py`** — Extração de texto de PDF com preservação de layout
- **`pdf/semantic_chunker.py`** — Chunking semântico com janela deslizante e sobreposição
- **`pdf/chunk_validator.py`** — Filtro de qualidade (contagem mínima/máxima de tokens, detecção de idioma)
- **`embedding/embedding_dispatcher.py`** — Roteia requisições para OpenAI ou sentence-transformers locais
- **`embedding/embedding_cache.py`** — Cache com suporte Redis indexado por hash de conteúdo
- **`embedding/vector_store.py`** — INSERT no pgvector e busca por similaridade cosseno

### Camada 4 — Camada de Workers (`src/carq/worker/`)
Processamento em segundo plano desacoplado da API:
- **`worker_pool.py`** — Pool de tarefas asyncio com concorrência configurável por estágio
- **`task_coordinator.py`** — Orquestra o pipeline de múltiplos estágios (parsear → chunkar → embutir → armazenar)

---

## Fluxo de Dados: Ingestão de Documentos

```
Client POST /embed
      │
      ├─ JWT verified
      ├─ Rate limiter token acquired (or 429 returned)
      │
      ▼
EmbeddingDispatcher
      │
      ├─ Cache hit? → return cached vector immediately
      │
      ├─ Circuit breaker OPEN? → fallback to local model or raise 503
      │
      └─ Call OpenAI text-embedding-3-large
             │
             ├─ Success → cache in Redis → INSERT into pgvector → return
             │
             └─ Failure → increment circuit breaker counter
                       → exponential backoff + retry (max 3)
                       → if exhausted → raise EmbeddingError
```

### Ingestão em Lote de PDF (Caminho do Worker)

```
Upload PDF
    │
    ├─ PDF Parser → extract raw text pages
    ├─ Semantic Chunker → sliding window (512 tokens, 64 overlap)
    ├─ Chunk Validator → filter quality chunks
    ├─ Embedding Dispatcher → batch embed (100 chunks/batch)
    └─ Vector Store → bulk INSERT with pgvector
```

---

## Padrões de Design Principais

| Padrão | Implementação | Finalidade |
|---|---|---|
| **Token Bucket** | `queue/rate_limiter.py` | Controlar o uso da API OpenAI dentro dos limites de RPM/TPM |
| **Circuit Breaker** | `queue/circuit_breaker.py` | Prevenir falhas em cascata quando a OpenAI está degradada |
| **Backoff Exponencial** | `queue/backoff_strategy.py` | Retry com jitter em falhas transitórias |
| **Cache-Aside** | `embedding/embedding_cache.py` | Cache Redis com TTL evita re-embedding de conteúdo idêntico |
| **Worker Pool** | `worker/worker_pool.py` | Concorrência limitada por estágio de processamento |
| **Health Probes** | `monitoring/health.py` | Sondas de vivacidade/prontidão do K8s; verifica latência do banco de dados e Redis |
| **Logging Estruturado** | `core/logging.py` | Logs JSON com IDs de correlação para rastreamento distribuído |

---

## Visão Geral do Esquema de Banco de Dados

```sql
-- Documents table
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    content_hash TEXT UNIQUE NOT NULL,   -- deduplication
    status      TEXT NOT NULL,            -- pending | processing | done | failed
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Chunks table
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Embeddings table (pgvector)
CREATE TABLE embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id    UUID REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    vector      vector(3072),             -- OpenAI text-embedding-3-large
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- IVFFlat index for fast approximate nearest-neighbor search
CREATE INDEX ON embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 100);
```

---

## Endpoints da API

| Método | Caminho | Descrição | Auth |
|---|---|---|---|
| `POST` | `/embed` | Gera embedding para texto | JWT |
| `POST` | `/embed/batch` | Embedding em lote de até 100 textos | JWT |
| `POST` | `/search` | Busca por similaridade semântica | JWT |
| `GET` | `/stats` | Estatísticas de fila e processamento | JWT |
| `GET` | `/health/live` | Sonda de vivacidade do Kubernetes | Nenhuma |
| `GET` | `/health/ready` | Sonda de prontidão do Kubernetes | Nenhuma |
| `GET` | `/metrics` | Exposição de métricas do Prometheus | Nenhuma |
