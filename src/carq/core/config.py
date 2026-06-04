"""Gerenciamento de configuração do CARQ via Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carrega o .env em os.environ para que todos os sub-modelos recebam os valores
# ao serem instanciados via default_factory (eles leem os.environ, não .env diretamente).
_env_file = Path(__file__).parents[3] / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)


class DatabaseSettings(BaseSettings):
    """Configurações de conexão com o PostgreSQL."""

    host: str = Field(default="localhost", description="Host do PostgreSQL")
    port: int = Field(default=5432, description="Porta do PostgreSQL")
    user: str = Field(default="postgres", description="Usuário do PostgreSQL")
    password: str = Field(default="postgres", description="Senha do PostgreSQL")
    database: str = Field(default="carq", description="Nome do banco de dados PostgreSQL")
    pool_size: int = Field(default=20, description="Tamanho do pool de conexões")
    max_overflow: int = Field(default=10, description="Máximo de conexões extras no pool")
    pool_timeout: int = Field(default=30, description="Timeout do pool em segundos")
    echo: bool = Field(default=False, description="Exibir queries SQL no log")

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    model_config = SettingsConfigDict(env_prefix="CARQ_DB_")


class RedisSettings(BaseSettings):
    """Configurações de conexão com o Redis (opcional para cache)."""

    enabled: bool = Field(default=False, description="Habilitar cache Redis")
    host: str = Field(default="localhost", description="Host do Redis")
    port: int = Field(default=6379, description="Porta do Redis")
    db: int = Field(default=0, description="Banco de dados do Redis")
    password: Optional[str] = Field(default=None, description="Senha do Redis")
    ttl: int = Field(default=3600, description="TTL do cache em segundos")

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(env_prefix="CARQ_REDIS_")


class RateLimiterSettings(BaseSettings):
    """Configuração do rate limiter."""

    # Limites da API de Embeddings da OpenAI
    openai_rpm: int = Field(default=3000, description="Requisições por minuto na OpenAI")
    openai_tpm: int = Field(default=1500000, description="Tokens por minuto na OpenAI")

    # Limites de embedding local
    local_embedding_workers: int = Field(
        default=4, description="Número de workers de embedding local"
    )

    # Parâmetros do token bucket
    refill_rate: float = Field(default=100.0, description="Tokens por segundo")
    burst_capacity: float = Field(default=1000.0, description="Capacidade máxima de burst")

    # Estratégia de backoff
    initial_backoff: float = Field(default=1.0, description="Backoff inicial em segundos")
    max_backoff: float = Field(default=300.0, description="Backoff máximo em segundos")
    exponential_base: float = Field(default=2.0, description="Base do backoff exponencial")

    model_config = SettingsConfigDict(env_prefix="CARQ_RATELIMIT_")


class CircuitBreakerSettings(BaseSettings):
    """Configuração do circuit breaker."""

    failure_threshold: int = Field(default=5, description="Falhas para abrir o circuito")
    recovery_timeout: int = Field(default=60, description="Timeout de recuperação em segundos")
    success_threshold: int = Field(default=2, description="Sucessos para fechar o circuito")

    model_config = SettingsConfigDict(env_prefix="CARQ_CIRCUIT_")


class WorkerSettings(BaseSettings):
    """Configuração do pool de workers."""

    pdf_parser_workers: int = Field(default=4, description="Workers de parser de PDF")
    chunk_workers: int = Field(default=8, description="Workers de chunking")
    embedding_workers: int = Field(default=2, description="Workers de embedding")
    vector_insert_workers: int = Field(default=4, description="Workers de inserção vetorial")

    batch_size: int = Field(default=100, description="Tamanho do lote para processamento")
    task_timeout: int = Field(default=300, description="Timeout de tarefa em segundos")
    retry_max_attempts: int = Field(default=3, description="Máximo de tentativas de retry")

    model_config = SettingsConfigDict(env_prefix="CARQ_WORKER_")


class EmbeddingSettings(BaseSettings):
    """Configuração do serviço de embedding."""

    provider: str = Field(default="openai", description="Provedor de embedding: openai|local")
    openai_api_key: Optional[str] = Field(default=None, description="Chave de API da OpenAI")
    openai_base_url: Optional[str] = Field(default=None, description="URL Base da OpenAI (para uso com endpoints compatíveis, ex: Mistral, vLLM)")
    openai_model: str = Field(
        default="text-embedding-3-large", description="Nome do modelo OpenAI"
    )

    local_model: str = Field(
        default="all-MiniLM-L6-v2", description="Modelo de embedding local (Hugging Face)"
    )
    local_device: str = Field(default="cpu", description="Dispositivo para embeddings locais")

    dimension: int = Field(default=3072, description="Dimensão do embedding")
    cache_embeddings: bool = Field(default=True, description="Cachear embeddings no Redis")

    model_config = SettingsConfigDict(env_prefix="CARQ_EMBEDDING_")


class APISettings(BaseSettings):
    """Configuração da API."""

    host: str = Field(default="0.0.0.0", description="Host da API")
    port: int = Field(default=8000, description="Porta da API")
    workers: int = Field(default=4, description="Workers do Uvicorn")
    debug: bool = Field(default=False, description="Modo debug")
    reload: bool = Field(default=False, description="Reload automático ao alterar código")

    cors_origins: list[str] = Field(
        default=["http://localhost:3000"], description="Origens permitidas pelo CORS"
    )
    jwt_secret: str = Field(default="dev-secret-change-in-prod", description="Segredo JWT")
    jwt_algorithm: str = Field(default="HS256", description="Algoritmo JWT")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Rejeita segredos JWT fracos ou obviamente inseguros."""
        weak_patterns = {
            "dev-secret",
            "change-in-prod",
            "password",
            "123456",
            "carq-dev-secret",
        }
        if len(v) < 32:
            raise ValueError(
                "jwt_secret must be at least 32 characters. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        lower = v.lower()
        if any(p in lower for p in weak_patterns):
            raise ValueError(
                "jwt_secret matches a known weak pattern. Use a cryptographically random value."
            )
        return v

    model_config = SettingsConfigDict(env_prefix="CARQ_API_")


class LoggingSettings(BaseSettings):
    """Configuração de logging."""

    level: str = Field(default="INFO", description="Nível de log")
    format: str = Field(default="json", description="Formato de log: json|text")
    file: Optional[str] = Field(default=None, description="Caminho do arquivo de log")

    model_config = SettingsConfigDict(env_prefix="CARQ_LOG_")


class Settings(BaseSettings):
    """Objeto de configuração raiz."""

    # Ambiente
    environment: str = Field(default="development", description="Nome do ambiente")
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("CARQ_DEBUG", "DEBUG", "debug"),
        description="Flag global de debug",
    )

    # Campos planos
    app_name: str = Field(
        default="CARQ",
        validation_alias=AliasChoices("CARQ_APP_NAME", "APP_NAME", "app_name"),
        description="Nome da aplicação",
    )
    api_base_url: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("CARQ_API_BASE_URL", "API_BASE_URL", "api_base_url"),
        description="URL base da API",
    )
    jwt_secret: str = Field(
        default="dev-secret-change-in-prod",
        validation_alias=AliasChoices("CARQ_JWT_SECRET", "JWT_SECRET", "jwt_secret"),
        description="Chave secreta JWT",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("CARQ_JWT_ALGORITHM", "JWT_ALGORITHM", "jwt_algorithm"),
        description="Algoritmo JWT",
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CARQ_OPENAI_API_KEY", "OPENAI_API_KEY", "openai_api_key"),
        description="Chave de API da OpenAI",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("CARQ_EMBEDDING_MODEL", "EMBEDDING_MODEL", "embedding_model"),
        description="Nome do modelo de embedding",
    )

    # Configurações de componentes
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rate_limiter: RateLimiterSettings = Field(default_factory=RateLimiterSettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Rastreamento
    enable_tracing: bool = Field(default=False, description="Habilitar rastreamento distribuído")
    jaeger_host: str = Field(default="localhost", description="Host do agente Jaeger")
    jaeger_port: int = Field(default=6831, description="Porta do agente Jaeger")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, v: str) -> str:
        allowed = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
        if v not in allowed:
            raise ValueError(f"JWT algorithm must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_production_safety(self) -> "Settings":
        """Previne configurações perigosas em produção."""
        if self.environment == "production" and self.api.reload:
            raise ValueError(
                "api.reload=True is not allowed in production (risk of RCE). "
                "Set CARQ_API_RELOAD=false."
            )
        if self.environment == "production":
            if not self.api.cors_origins or self.api.cors_origins == ["http://localhost:3000"]:
                raise ValueError(
                    "cors_origins must be explicitly configured in production. "
                    "Set CARQ_API_CORS_ORIGINS to your actual frontend origins."
                )
            if "*" in self.api.cors_origins:
                raise ValueError(
                    "Wildcard CORS origin ('*') is not allowed in production. "
                    "Set CARQ_API_CORS_ORIGINS to specific allowed origins."
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
