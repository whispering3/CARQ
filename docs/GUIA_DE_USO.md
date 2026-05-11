# CARQ — Guia de Uso

> **Context-Aware RAG Processing Queue**  
> Fila assíncrona de ingestão de documentos para pipelines de RAG (Retrieval-Augmented Generation).

---

## O que é o CARQ?

O CARQ é um servidor que recebe PDFs (ou texto), processa cada um de forma assíncrona e gera **embeddings vetoriais** armazenados no PostgreSQL com pgvector. Esses vetores podem ser consultados por similaridade semântica — a base de qualquer sistema RAG.

O diferencial é que ele **não derruba o pipeline inteiro quando algo falha**. Cada chunk de cada documento é rastreado individualmente. Se a API da OpenAI retornar 429, o sistema espera e tenta de novo. Se um chunk específico falhar três vezes, só ele vai para a fila de mortos (DLQ) — o restante continua.

```
Você envia um PDF → CARQ fatia em chunks → gera embeddings → guarda vetores
                                                ↓ falha?
                                   Retenta com backoff exponencial
                                                ↓ ainda falha?
                                   Manda para a Dead Letter Queue (DLQ)
```

---

## O que você precisa ter instalado

| Dependência | Versão mínima | Obrigatório? |
|-------------|--------------|--------------|
| Python | 3.11 | ✅ Sim |
| PostgreSQL | 15 com `pgvector` | ✅ Sim |
| Redis | 6 | ⚠️ Recomendado (cache + rate limit distribuído) |
| Docker + Compose | qualquer | Opcional (simplifica o setup) |

---

## Configuração — passo a passo

### 1. Instale as dependências Python

```bash
# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Instale o projeto
pip install -e ".[dev]"
```

### 2. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com as informações do seu ambiente. As variáveis mínimas para funcionar:

```env
# Banco de dados
CARQ_DB_HOST=localhost
CARQ_DB_PORT=5432
CARQ_DB_USER=postgres
CARQ_DB_PASSWORD=sua-senha-aqui
CARQ_DB_DATABASE=carq

# Chave de API da OpenAI (obrigatória para embeddings)
CARQ_EMBEDDING_OPENAI_API_KEY=sk-...

# Chave(s) de acesso à API do CARQ
# Gere com: python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"
CARQ_API_KEYS=sk-minha-chave

# Segredo JWT (mínimo 32 caracteres, aleatório)
# Gere com: python -c "import secrets; print(secrets.token_hex(32))"
CARQ_API_JWT_SECRET=cole-aqui-um-valor-aleatorio-de-64-chars
```

### 3. Suba o PostgreSQL e Redis com Docker (opcional)

```bash
docker-compose up -d
```

Isso sobe o PostgreSQL na porta **5433** e o Redis na porta **6380** (portas alternativas para não conflitar com instalações locais).

### 4. Rode as migrações

```bash
alembic upgrade head
```

Isso cria todas as tabelas e instala a extensão `pgvector`.

### 5. Inicie o servidor

```bash
uvicorn carq.main:app --host 0.0.0.0 --port 8000
```

Você deve ver:

```
INFO: CARQ is ready.
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Como usar — CLI

O CARQ vem com uma CLI completa. Antes de usá-la, defina as variáveis de conexão:

```bash
export CARQ_API_URL=http://localhost:8000
export CARQ_API_KEY=sk-minha-chave
```

### Comandos disponíveis

#### `carq ingest` — Enviar um documento para processamento

```bash
# Envio simples
python -m carq.cli.client ingest meu_documento.pdf

# Envio com tipo, prioridade e metadados
python -m carq.cli.client ingest relatorio.pdf \
  --document-type pdf \
  --priority 10 \
  --metadata '{"categoria": "financeiro", "ano": 2025}'

# Envio e aguarda a conclusão na mesma linha de comando
python -m carq.cli.client ingest meu_documento.pdf --wait
```

**O que esperar:** O servidor retorna um `document_id` imediatamente. O processamento acontece em background — pode levar de segundos a minutos dependendo do tamanho do documento e do número de chunks.

---

#### `carq status` — Consultar o status de um documento

```bash
python -m carq.cli.client status <document_id>
```

**Saída exemplo:**

```
┌───────────────────────────────────────────┐
│ Document abc123...                        │
├─────────────┬─────────────────────────────┤
│ Field       │ Value                       │
│ status      │ done                        │
│ chunks      │ 42                          │
│ embeddings  │ 42                          │
│ created_at  │ 2026-05-10 22:00:00         │
└─────────────┴─────────────────────────────┘
```

**Status possíveis:**

| Status | Significado |
|--------|-------------|
| `pending` | Aguardando na fila |
| `processing` | Em processamento |
| `done` | Processado com sucesso — embeddings disponíveis |
| `failed` | Falhou após todas as tentativas → DLQ |

---

#### `carq list` — Listar documentos

```bash
# Todos os documentos
python -m carq.cli.client list

# Filtrado por status
python -m carq.cli.client list --status-filter failed

# Com limite
python -m carq.cli.client list --limit 50

# Saída em JSON
python -m carq.cli.client list --output json
```

---

#### `carq retry` — Reprocessar um documento com falha

```bash
python -m carq.cli.client retry <document_id>
```

Coloca o documento de volta na fila com status `pending`. Use quando a causa da falha já foi corrigida (ex.: API key inválida, banco fora do ar).

---

#### `carq stats` — Estatísticas do sistema

```bash
python -m carq.cli.client stats
```

Mostra totais de documentos processados, chunks, embeddings, taxa de erros e uso de tokens.

---

#### `carq health` — Verificar se o sistema está saudável

```bash
python -m carq.cli.client health
```

Retorna `✅` se banco e Redis estão respondendo, `❌` se algo estiver fora.

---

## Como usar — API REST

A documentação interativa fica em `http://localhost:8000/docs` (Swagger UI).

### Autenticação

Todas as rotas requerem um header de autenticação:

```
X-API-Key: sk-minha-chave
```

### Enviar um documento

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-API-Key: sk-minha-chave" \
  -H "Content-Type: application/json" \
  -d '{
    "source_uri": "file:///caminho/para/relatorio.pdf",
    "document_type": "pdf",
    "priority": 0,
    "attributes": {"categoria": "financeiro"}
  }'
```

**Resposta:**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Document accepted for processing"
}
```

### Consultar status

```bash
curl http://localhost:8000/api/v1/documents/550e8400-... \
  -H "X-API-Key: sk-minha-chave"
```

### Busca semântica (após processamento)

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: sk-minha-chave" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quais são os riscos de crédito mencionados?",
    "limit": 5,
    "similarity_threshold": 0.7
  }'
```

---

## O que esperar em termos de desempenho

| Cenário | Tempo estimado |
|---------|---------------|
| PDF pequeno (< 10 páginas) | 5–30 segundos |
| PDF médio (10–100 páginas) | 30 segundos – 5 minutos |
| PDF grande (100+ páginas) | 5–20 minutos |
| Documento já no cache Redis | < 1 segundo por chunk |

**Fatores que afetam o tempo:**
- Tamanho e complexidade do PDF
- Limites de rate da API OpenAI (`CARQ_RATELIMIT_OPENAI_RPM`)
- Número de workers configurados
- Latência da rede até a OpenAI

**O CARQ nunca derruba outros documentos** por causa de um lento. Cada documento roda em paralelo com os demais dentro dos limites configurados.

---

## Comportamento esperado frente a erros

### API OpenAI retornou 429 (rate limit)

O CARQ detecta automaticamente, pausa o worker correspondente com backoff exponencial (começa em 1s, vai até 5 minutos) e retoma sem intervenção. Você verá nos logs:

```json
{"level": "WARNING", "message": "Rate limit hit, backing off 4.0s", "provider": "openai"}
```

### Documento falhou após 3 tentativas

O documento vai para a **Dead Letter Queue** (DLQ) com status `failed`. Use `carq retry <id>` para recolocar na fila após corrigir a causa.

### Banco de dados ficou indisponível

O circuit breaker abre após 5 falhas consecutivas, rejeitando novas requisições com `503` por 60 segundos, depois testa novamente. O Kubernetes readiness probe (`/health/ready`) vai reportar `not_ready` e o balanceador para de enviar tráfego para a instância afetada.

### Chunk inválido (muito pequeno, truncado, duplicado)

O `ChunkValidator` descarta automaticamente chunks de baixa qualidade antes de enviar para a OpenAI. Isso é registrado nos logs mas **não falha o documento** — os chunks válidos continuam sendo processados.

---

## Monitoramento

### Endpoints de saúde

| Endpoint | Uso | O que verifica |
|----------|-----|----------------|
| `GET /health/live` | Kubernetes liveness probe | Processo está vivo |
| `GET /health/ready` | Kubernetes readiness probe | Banco + Redis respondendo |
| `GET /health` | Dashboard humano | Status detalhado |

### Métricas Prometheus

```bash
curl http://localhost:8000/metrics
```

Métricas principais:

| Métrica | O que mede |
|---------|-----------|
| `carq_http_requests_total` | Total de requisições por endpoint/status |
| `carq_http_request_duration_seconds` | Latência de cada endpoint |
| `carq_chunks_processed_total` | Chunks processados com sucesso |
| `carq_embedding_cost_usd_total` | Custo acumulado em USD de embeddings |
| `carq_task_queue_depth` | Documentos aguardando na fila |

### Logs estruturados (JSON)

Todos os logs são em JSON com `correlation_id` para rastrear o fluxo completo de um documento:

```bash
# Seguir logs em produção e filtrar por documento específico
kubectl logs -f deployment/carq | grep "document_id\":\"550e8400"
```

---

## Limites e restrições padrão

| Limite | Valor padrão | Variável para alterar |
|--------|-------------|----------------------|
| Requisições por cliente por minuto | 600 | Código em `main.py` (`rpm=600`) |
| Tamanho máximo de arquivo | 50 MB | `CARQ_API_MAX_FILE_SIZE` |
| Timeout por tarefa | 300 segundos | `CARQ_WORKER_TASK_TIMEOUT` |
| Tentativas antes do DLQ | 3 | `CARQ_WORKER_RETRY_MAX_ATTEMPTS` |
| Chunks máximos por documento | 10.000 | Configurável em `ChunkingConfig` |
| Tokens mínimos por chunk | 10 | Configurável em `ChunkValidator` |

---

## Problemas comuns e soluções

### `Connection refused` ao iniciar

**Causa:** PostgreSQL não está rodando ou está na porta errada.  
**Solução:** `docker-compose up -d` ou verifique `CARQ_DB_HOST` e `CARQ_DB_PORT` no `.env`.

### `jwt_secret must be at least 32 characters`

**Causa:** O `CARQ_API_JWT_SECRET` no `.env` é muito curto ou é o valor padrão.  
**Solução:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
# Cole o resultado em CARQ_API_JWT_SECRET no .env
```

### Documento fica em `processing` para sempre

**Causa:** Worker travou ou processo foi encerrado durante o processamento.  
**Solução:** O `stuck_task_recovery_loop` detecta tarefas paradas há mais de `CARQ_WORKER_TASK_TIMEOUT` segundos e as recoloca na fila automaticamente. Se quiser forçar: `carq retry <id>`.

### Embeddings com qualidade baixa nos resultados de busca

**Causa provável:** `similarity_threshold` muito baixo (retorna resultados não relacionados) ou chunks muito grandes/pequenos.  
**Solução:** Ajuste `similarity_threshold` para `0.75–0.85` na busca e revise `CARQ_WORKER_CHUNK_*` para chunks adequados ao seu conteúdo.

### `Rate limit exceeded` ao chamar a API do CARQ

**Causa:** Mais de 600 requisições por minuto do mesmo IP/API-key.  
**Solução:** Distribua as chamadas no tempo ou aumente o limite em `main.py` (`rpm=600`). Em produção com Redis habilitado, o limite é compartilhado entre todos os workers (distribuído).

---

## Variáveis de ambiente — referência rápida

```env
# Infraestrutura
CARQ_DB_HOST, CARQ_DB_PORT, CARQ_DB_USER, CARQ_DB_PASSWORD, CARQ_DB_DATABASE
CARQ_REDIS_ENABLED, CARQ_REDIS_HOST, CARQ_REDIS_PORT, CARQ_REDIS_PASSWORD

# Segurança
CARQ_API_KEYS=sk-chave1,sk-chave2        # Chaves de acesso à API
CARQ_API_JWT_SECRET=<64 hex chars>        # Segredo JWT

# Embeddings
CARQ_EMBEDDING_PROVIDER=openai            # openai | local
CARQ_EMBEDDING_OPENAI_API_KEY=sk-...
CARQ_EMBEDDING_OPENAI_MODEL=text-embedding-3-large
CARQ_EMBEDDING_DIMENSION=3072

# Workers
CARQ_WORKER_PDF_PARSER_WORKERS=4
CARQ_WORKER_CHUNK_WORKERS=8
CARQ_WORKER_EMBEDDING_WORKERS=2
CARQ_WORKER_RETRY_MAX_ATTEMPTS=3
CARQ_WORKER_TASK_TIMEOUT=300

# Rate limiting (OpenAI)
CARQ_RATELIMIT_OPENAI_RPM=3000
CARQ_RATELIMIT_OPENAI_TPM=1500000

# Ambiente
CARQ_ENVIRONMENT=development              # development | staging | production
CARQ_LOG_LEVEL=INFO
CARQ_LOG_FORMAT=json
```

> Veja o arquivo `.env.example` na raiz do projeto para a lista completa com comentários.
