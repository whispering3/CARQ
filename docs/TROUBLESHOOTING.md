# Guia de Resolução de Problemas

Este guia abrange os problemas mais comuns do CARQ em ambientes locais, de CI e Kubernetes.

## 1. A API não inicia

### Sintomas
- Uvicorn falha na inicialização
- `/health/live` está inacessível

### Verificações
1. Confirme que as dependências estão instaladas:
   ```bash
   pip install -e ".[dev]"
   ```
2. Confirme que as variáveis de ambiente necessárias estão definidas (`.env` ou ambiente):
   - `CARQ_DB_HOST`, `CARQ_DB_PORT`, `CARQ_DB_USER`, `CARQ_DB_PASSWORD`, `CARQ_DB_DATABASE`
   - `CARQ_API_KEYS` (ou fallback de desenvolvimento válido)
   - `CARQ_EMBEDDING_OPENAI_API_KEY`
3. Execute:
   ```bash
   uvicorn carq.main:app --reload
   ```

## 2. Erros de autenticação (401 / 503)

### 401 Chave de API inválida
- Certifique-se de que a requisição inclui `X-API-Key` com um dos valores em `CARQ_API_KEYS`.

### 503 Repositório de chaves de API não configurado
- Configure:
  ```env
  CARQ_API_KEYS=your-key-1,your-key-2
  ```
- Reinicie a API após alterar as variáveis de ambiente.

## 3. Falhas de conexão com o banco de dados

### Sintomas
- `/health/ready` retorna não saudável
- Erros de conexão do SQLAlchemy

### Verificações
1. Confirme que o PostgreSQL está em execução e acessível.
2. Confirme que a extensão `pgvector` existe:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Aplique as migrações:
   ```bash
   alembic upgrade head
   ```

## 4. Incompatibilidades de migração/esquema

### Sintomas
- Erros de tempo de execução por colunas ausentes
- Falhas de inserção/busca em embeddings

### Correção
1. Re-execute as migrações no banco de dados alvo.
2. Verifique se o esquema inclui:
   - `rag_embeddings.text`
   - `rag_embeddings.tokens_used`
   - `rag_embeddings.cost_usd`
3. Valide a revisão atual:
   ```bash
   alembic current
   ```

## 5. Falhas de embedding e OpenAI

### Sintomas
- `/api/v1/embed` ou `/api/v1/search` retorna 500

### Verificações
1. Verifique `CARQ_EMBEDDING_OPENAI_API_KEY`.
2. Verifique os limites de taxa e a cota do provedor.
3. Inspecione os logs em busca de `EmbeddingError` e comportamento de retry/backoff.

## 6. Tarefas do worker presas na fila

### Sintomas
- Documentos permanecem como `pending`
- Tarefas permanecem como `processing` por longos períodos

### Verificações
1. Confirme que o processo de worker/coordenador está em execução.
2. Verifique o status/tentativas em `processing_tasks`.
3. Verifique as configurações de recuperação de tarefas presas na configuração do coordenador.

## 7. Crescimento do DLQ / falhas repetidas de tarefas

### Sintomas
- Muitas tarefas marcadas para DLQ

### Ações
1. Inspecione `attributes.error_details` da tarefa.
2. Corrija a causa raiz (parser, embedding, banco de dados, rede).
3. Replique tarefas do DLQ somente após validar a correção.

## 8. Endpoint de métricas ausente ou vazio

### Sintomas
- `/metrics` retorna vazio ou está indisponível

### Verificações
1. Confirme que a API foi iniciada com o middleware de métricas habilitado.
2. Certifique-se de que o Prometheus consegue alcançar o serviço da API.
3. Valide se o caminho de coleta é `/metrics`.

## 9. Falhas no pipeline de CI/CD

### Causas comuns
- Erros de lint/tipagem
- Resultados da varredura de segurança
- Segredos do GitHub ausentes para implantação

### Verificações
1. Execute localmente antes de enviar:
   ```bash
   pytest -q
   ```
2. Verifique se os segredos estão configurados:
   - `KUBECONFIG_STAGING`
   - `KUBECONFIG_PROD`
   - Segredos de registro/autenticação conforme necessário

## 10. Problemas no rollout do Kubernetes

### Sintomas
- CrashLoopBackOff
- Falhas na sonda de prontidão

### Verificações
1. Inspecione os logs do pod:
   ```bash
   kubectl logs -f deployment/carq-api -n carq
   ```
2. Descreva os eventos do pod:
   ```bash
   kubectl describe pod <pod-name> -n carq
   ```
3. Valide as sondas:
   - Vivacidade: `/health/live`
   - Prontidão: `/health/ready`

## Lista de verificação rápida

1. `GET /health/live` retorna 200.
2. `GET /health/ready` retorna saudável.
3. `GET /metrics` retorna métricas do Prometheus.
4. `POST /api/v1/embed` tem sucesso com `X-API-Key` válido.
5. `POST /api/v1/documents` aceita ingestão e atualizações de status.
