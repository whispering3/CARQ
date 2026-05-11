# Runbooks CARQ

Procedimentos operacionais para o sistema CARQ em produção.

---

## 1. Inicializando o Sistema

### Desenvolvimento Local
```bash
# Sobe toda a stack (PostgreSQL + pgvector + Redis + API)
docker compose -f docker/docker-compose.yml up -d

# Aplica as migrações do banco
docker compose -f docker/docker-compose.yml exec api \
  alembic -c migrations/alembic.ini upgrade head

# Verifica a saúde do serviço
curl http://localhost:8000/health/ready
```

### Kubernetes (Produção)
```bash
# 1. Aplica todos os manifestos em ordem
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# 2. Cria os secrets (edite k8s/secret.yaml com os valores base64 reais primeiro)
kubectl apply -f k8s/secret.yaml

# 3. Implanta os workloads
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Executa as migrações do banco como Job avulso
kubectl run carq-migrate --rm -i --restart=Never \
  --image=ghcr.io/<org>/carq:latest \
  -n carq \
  --env-from=secret/carq-secret \
  --env-from=configmap/carq-config \
  -- alembic -c migrations/alembic.ini upgrade head

# 5. Verifica o rollout
kubectl rollout status deployment/carq-api -n carq
kubectl rollout status deployment/carq-worker -n carq
kubectl get pods -n carq
```

---

## 2. Escalando Workers

### Escala manual (ex.: antes de um grande lote de ingestão)
```bash
# Aumenta réplicas da API
kubectl scale deployment/carq-api --replicas=6 -n carq

# Aumenta réplicas dos workers
kubectl scale deployment/carq-worker --replicas=10 -n carq

# Reduz novamente após o lote concluir
kubectl scale deployment/carq-worker --replicas=4 -n carq
```

### HPA está ativo — verificar status atual
```bash
kubectl get hpa -n carq
kubectl describe hpa carq-api-hpa -n carq
```

### Ajustar limites do HPA sem editar o YAML
```bash
kubectl patch hpa carq-api-hpa -n carq \
  -p '{"spec":{"maxReplicas":30}}'
```

---

## 3. Monitoramento com Prometheus & Grafana

### Principais métricas Prometheus expostas em `/metrics`

| Métrica | Tipo | Descrição |
|---|---|---|
| `carq_requests_total` | Contador | Total de requisições HTTP por método/caminho/status |
| `carq_request_duration_seconds` | Histograma | Latência por endpoint |
| `carq_embeddings_generated_total` | Contador | Embeddings produzidos por provedor |
| `carq_embedding_cache_hits_total` | Contador | Acertos no cache Redis |
| `carq_circuit_breaker_state` | Gauge | 0=fechado, 1=aberto, 2=semi-aberto |
| `carq_queue_depth` | Gauge | Tarefas aguardando na fila |
| `carq_worker_active` | Gauge | Workers ativos por estágio |
| `carq_rate_limiter_tokens` | Gauge | Capacidade restante do token bucket |

### Painéis do Dashboard Grafana (recomendados)
1. **Taxa de Requisições** — `rate(carq_requests_total[5m])`
2. **Latência P99** — `histogram_quantile(0.99, rate(carq_request_duration_seconds_bucket[5m]))`
3. **Taxa de Erros** — `rate(carq_requests_total{status=~"5.."}[5m])`
4. **Estado do Circuit Breaker** — `carq_circuit_breaker_state`
5. **Profundidade da Fila** — `carq_queue_depth`
6. **Taxa de Acerto do Cache** — `rate(carq_embedding_cache_hits_total[5m]) / rate(carq_embeddings_generated_total[5m])`

### Alertas Recomendados

```yaml
# Regras de alerta do Prometheus
groups:
  - name: carq
    rules:
      - alert: CarqHighErrorRate
        expr: rate(carq_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Taxa de erros do CARQ > 5%"

      - alert: CarqCircuitBreakerOpen
        expr: carq_circuit_breaker_state == 1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker do CARQ ABERTO — chamadas à OpenAI falhando"

      - alert: CarqQueueDepthHigh
        expr: carq_queue_depth > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Fila do CARQ com mais de 1000 tarefas"
```

---

## 4. Backup e Restauração do Banco de Dados

### Backup (pg_dump)
```bash
# A partir de um pod temporário no cluster
kubectl run pg-backup --rm -i --restart=Never \
  --image=postgres:16 \
  -n carq \
  -- pg_dump \
    -h postgres-svc \
    -U postgres \
    -d carq \
    -F c \
    -f /dev/stdout \
  > carq_backup_$(date +%Y%m%d_%H%M%S).pgc

# Em produção, use um CronJob ou backup gerenciado (RDS, Cloud SQL snapshots)
```

### Restaurar
```bash
# Para os workers primeiro para evitar escritas durante a restauração
kubectl scale deployment/carq-worker --replicas=0 -n carq

kubectl run pg-restore --rm -i --restart=Never \
  --image=postgres:16 \
  -n carq \
  -- pg_restore \
    -h postgres-svc \
    -U postgres \
    -d carq \
    --clean \
    --if-exists \
    /path/to/backup.pgc

# Reinicia os workers
kubectl scale deployment/carq-worker --replicas=4 -n carq
```

### Recuperação em Ponto no Tempo
Configure o arquivamento WAL no PostgreSQL e use `pg_basebackup` para PITR. Em bancos de dados gerenciados na nuvem (RDS, Cloud SQL), habilite backups automáticos e use o console para PITR.

---

## 5. Procedimentos de Resposta a Incidentes

### Níveis de Severidade
| Nível | Descrição | SLA |
|---|---|---|
| P1 | Interrupção total — nenhuma requisição atendida | Resposta em 15 min |
| P2 | Degradação parcial — taxa de erros > 5% | Resposta em 30 min |
| P3 | Desempenho degradado — latência > 2x | Resposta em 2 horas |
| P4 | Problema menor — somente alertas de monitoramento | Próximo dia útil |

### Runbook de Resposta P1
```bash
# 1. Verifica saúde dos pods
kubectl get pods -n carq
kubectl describe pod <pod-com-crash> -n carq

# 2. Verifica logs recentes
kubectl logs -l app=carq -n carq --since=10m --prefix

# 3. Verifica eventos do cluster
kubectl get events -n carq --sort-by='.lastTimestamp'

# 4. Se o banco estiver inacessível — faz rollback para o último deploy estável
kubectl rollout undo deployment/carq-api -n carq
kubectl rollout undo deployment/carq-worker -n carq

# 5. Verifica se o rollback concluiu
kubectl rollout status deployment/carq-api -n carq
```

### Caminho de Escalação
1. Engenheiro de plantão (PagerDuty)
2. Engenheiro líder → administrador de banco de dados
3. Suporte do provedor de nuvem

---

## 6. Esvaziamento Gracioso da Fila

Use antes de manutenção planejada ou redução do número de workers.

```bash
# Passo 1: Para de aceitar novas tarefas pausando o ingress
kubectl annotate ingress carq-ingress -n carq \
  nginx.ingress.kubernetes.io/server-snippet='return 503 "Maintenance";'

# Passo 2: Monitora a profundidade da fila até esvaziar
watch -n 5 'kubectl exec -n carq \
  $(kubectl get pod -l app=carq,component=api -n carq -o name | head -1) \
  -- curl -sf http://localhost:8000/stats | python3 -c "import sys,json; d=json.load(sys.stdin); print(\"Fila:\", d[\"queue_depth\"])"'

# Passo 3: Reduz os workers quando a fila estiver vazia
kubectl scale deployment/carq-worker --replicas=0 -n carq

# Passo 4: Realiza a manutenção ...

# Passo 5: Retoma — sobe os workers e remove a anotação 503
kubectl scale deployment/carq-worker --replicas=4 -n carq
kubectl annotate ingress carq-ingress -n carq \
  nginx.ingress.kubernetes.io/server-snippet-
```

---

## 7. Atualização de Imagem Progressiva (Manual)

```bash
# Atualiza a imagem da API para um SHA específico
kubectl set image deployment/carq-api \
  carq=ghcr.io/<org>/carq:<sha> \
  -n carq

# Acompanha o rollout
kubectl rollout status deployment/carq-api -n carq --timeout=5m

# Faz rollback se necessário
kubectl rollout undo deployment/carq-api -n carq

# Visualiza o histórico de rollouts
kubectl rollout history deployment/carq-api -n carq
```

---

## 8. Validação de Carga (Baseline Sintético Local)

O baseline a seguir foi executado localmente com os runners sintéticos do `tests/load/load_test_python.py`:

- **Carga sintética de embedding** (`50 usuários`, `30s`)
  - Throughput: **~800,89 req/s**
  - Latência P99: **~76,20 ms**
  - Taxa de erros: **0%**

- **Carga sintética de ingestão** (`20 usuários`, `60s`)
  - Throughput: **~180,55 req/s** (equivalente a ~10.833+ req/min)
  - Latência P99: **~136,59 ms**
  - Taxa de erros: **0%**

Esses números validam a lógica central de throughput para o caminho sintético e superam a meta de referência de 10k chunks/min ao usar as premissas atuais de simulação de ingestão.
