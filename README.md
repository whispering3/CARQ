# CARQ - Context-Aware RAG Processing Queue

Sistema de orquestração de nível empresarial para pipelines de ingestão de documentos RAG, projetado para alta disponibilidade, resiliência a falhas e limitação de taxa adaptativa.

[![Mentioned in Awesome Data Engineering](https://awesome.re/mentioned-badge.svg)](https://github.com/igorbarinov/awesome-data-engineering)
[![Docker Pulls](https://img.shields.io/docker/pulls/choquelnews/carq.svg)](https://hub.docker.com/r/choquelnews/carq)
[![GitHub Release](https://img.shields.io/github/v/release/whispering3/CARQ.svg)](https://github.com/whispering3/CARQ/releases)

## Visão Geral

**CARQ** resolve três problemas críticos de engenharia que impedem pipelines RAG ingênuos de escalar:

1. **Explosão de Limite de Taxa** - Tratamento elegante dos limites de taxa de API sem falhas em cascata catastróficas
2. **Reprocessamento Catastrófico** - Recuperação cirúrgica de falhas (apenas os chunks com falha são retentados, não documentos inteiros)
3. **Contenção de Recursos** - Pooling adaptativo e coordenação de workers para evitar saturação de conexões

## Principais Funcionalidades

✅ **Throughput**: 10.000+ chunks/minuto com degradação elegante  
✅ **Durabilidade**: Garantias ACID via PostgreSQL (zero brokers externos)  
✅ **Resiliência**: Adaptação automática de limite de taxa, circuit breaker, fila de mensagens mortas  
✅ **Idempotência**: Deduplicação baseada em hash de conteúdo evita reprocessamento  
✅ **Observabilidade**: Logging JSON estruturado, métricas Prometheus, IDs de correlação  
✅ **Escalonamento Horizontal**: Coordenação de workers multi-instância via PostgreSQL

## Arquitetura

```text
┌─────────────────────────────────┐
│    ENTRY LAYER                  │  REST API, CLI, S3 Events
├─────────────────────────────────┤
│  ORCHESTRATION LAYER            │  Queue Manager, Rate Limiter, Circuit Breaker
├─────────────────────────────────┤
│    WORKER LAYER                 │  PDF Parser, Semantic Chunker, Embedding Dispatcher
├─────────────────────────────────┤
│  PERSISTENCE LAYER              │  PostgreSQL + pgvector (zero external brokers)
└─────────────────────────────────┘

```

### Schema do Banco de Dados

Entidades principais:

* **rag_documents**: Documentos raiz (PDFs, URLs, texto bruto)
* **rag_chunks**: Segmentos de documentos fragmentados com metadados
* **rag_embeddings**: Embeddings vetoriais com suporte a busca semântica
* **processing_tasks**: Fila de tarefas assíncronas com rastreamento de status
* **task_deadletter**: Tarefas com falha para investigação manual

## Stack Tecnológica

* **Runtime**: Python 3.11+
* **Framework**: FastAPI + asyncio
* **Banco de Dados**: PostgreSQL 15+ com extensão pgvector
* **ORM**: SQLAlchemy 2.0
* **Async**: aiohttp, asyncio-contextmanager
* **Testes**: pytest, pytest-asyncio
* **Deploy**: Docker, Kubernetes

## Instalação

### Pré-requisitos

* Python 3.11+
* PostgreSQL 15+ (com extensão pgvector)
* Redis 6+ (opcional, para cache de embeddings)

### Configuração para Desenvolvimento Local

```bash
# Clonar repositório
git clone [https://github.com/whispering3/CARQ.git](https://github.com/whispering3/CARQ.git)
cd carq

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -e ".[dev]"

# Configurar ambiente
cp .env.example .env
# Edite .env com sua configuração

# Executar migrações
alembic upgrade head

# Iniciar o servidor
uvicorn carq.api.server:app --reload

```

## Início Rápido

### 1. Criar um Job de Ingestão de Documento

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_uri": "s3://bucket/path/to/document.pdf",
    "metadata": {
      "category": "technical",
      "language": "en"
    }
  }'

```

### 2. Monitorar Progresso

```bash
curl http://localhost:8000/api/v1/tasks/{task_id}/status

```

### 3. Consultar Documentos Processados

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to configure rate limiting",
    "top_k": 5,
    "similarity_threshold": 0.7
  }'

```

## Configuração

Todas as configurações utilizam variáveis de ambiente (veja `.env.example`):

```env
# Banco de Dados
CARQ_DB_HOST=localhost
CARQ_DB_PORT=5432
CARQ_DB_DATABASE=carq

# Embeddings
CARQ_EMBEDDING_PROVIDER=openai
CARQ_EMBEDDING_OPENAI_API_KEY=sk-...

# Workers
CARQ_WORKER_PDF_PARSER_WORKERS=4
CARQ_WORKER_EMBEDDING_WORKERS=2

# Limitação de Taxa
CARQ_RATELIMIT_OPENAI_RPM=3000
CARQ_RATELIMIT_OPENAI_TPM=1500000

```

## Pipeline de Processamento

1. **Ingestão**: Documento enviado via API/CLI/evento S3
2. **Enfileiramento**: Tarefa enfileirada com status `pending` em `processing_tasks`
3. **Parsing de PDF**: Extração assíncrona de PDF com isolamento de erros
4. **Chunking**: Segmentação semântica com preservação de contexto
5. **Embedding**: API OpenAI ou modelo local com limitação de taxa adaptativa
6. **Persistência**: Inserção vetorial com segurança transacional
7. **Conclusão**: Status atualizado, metadados indexados

## Tratamento de Erros

O CARQ utiliza recuperação de erros estruturada:

* **Limite de Taxa (429)**: Backoff automático, pausa da fila por worker
* **Erros Transitórios (5xx)**: Backoff exponencial com jitter, máximo de 3 tentativas
* **Erros de Parsing**: Tarefa movida para DLQ, investigação do operador necessária
* **Inserção Vetorial**: Rollback de transação, reprocessamento do chunk

## Monitoramento

### Endpoints de Saúde

```bash
# Probe de vivacidade (saúde do servidor)
curl http://localhost:8000/health/live

# Probe de prontidão (todas as dependências prontas)
curl http://localhost:8000/health/ready

# Status detalhado
curl http://localhost:8000/health/status

```

### Métricas (Prometheus)

```bash
curl http://localhost:8000/metrics

```

Métricas principais:

* `carq_chunks_processed_total` - Total de chunks processados
* `carq_embedding_latency_seconds` - Latência de geração de embeddings
* `carq_task_queue_depth` - Tarefas pendentes na fila
* `carq_rate_limiter_backoff_seconds` - Duração atual do backoff

## Testes

```bash
# Testes unitários
pytest tests/unit -v

# Testes de integração (requer PostgreSQL)
pytest tests/integration -v

# Testes de carga (10.000+ chunks/min)
pytest tests/load -v --benchmark-only

# Relatório de cobertura
pytest --cov=carq --cov-report=html

```

## Docker

Para executar o projeto rapidamente em produção ou testes isolados, utilize a imagem oficial:

```bash
# Baixar a imagem oficial
docker pull choquelnews/carq:1

# Executar via docker-compose (Inicia o Postgres, Redis e CARQ)
docker-compose up -d

# Verificar logs
docker-compose logs -f carq

```

## Deploy no Kubernetes

```bash
# Criar namespace
kubectl create namespace carq

# Fazer deploy
kubectl apply -f k8s/ -n carq

# Verificar rollout
kubectl rollout status deployment/carq -n carq

# Ver logs
kubectl logs -f deployment/carq -n carq

```

## Documentação

* [Arquitetura e Design](https://www.google.com/search?q=docs/ARCHITECTURE.md)
* [Referência da API](https://www.google.com/search?q=docs/API.md)
* [Schema do Banco de Dados](https://www.google.com/search?q=docs/SCHEMA.md)
* [Runbooks Operacionais](https://www.google.com/search?q=docs/RUNBOOKS.md)

## 🤝 Contribuindo

1. Crie um branch de funcionalidade (`git checkout -b feature/minha-feature`)
2. Faça commit das alterações (`git commit -am 'Adicionar funcionalidade'`)
3. Envie para o branch (`git push origin feature/minha-feature`)
4. Crie um Pull Request

## 📝 Licença

Licença MIT - veja o arquivo LICENSE

## 📞 Suporte

* Issues: https://github.com/whispering3/CARQ/issues
* Discussões: https://github.com/whispering3/CARQ/discussions
* E-mail: drsouza14@gmail.com

---

**Status**: Produção (v1.0.0)

**Última Atualização**: Maio de 2026
