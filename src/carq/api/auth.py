"""Autenticação da API via chave X-API-Key e Bearer token."""

import logging
import os
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, status, Header

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_valid_keys() -> frozenset[str]:
    """Carrega as chaves de API válidas a partir do ambiente.

    As chaves são lidas de CARQ_API_KEYS (separadas por vírgula).
    Se não estiver definido, a aplicação registra um aviso crítico e bloqueia a API
    (retorna um conjunto vazio, fazendo todas as requisições receberem 503).

    Nunca usa o segredo JWT como fallback — as duas credenciais devem ser independentes.
    """
    raw = os.environ.get("CARQ_API_KEYS", "").strip()
    if raw:
        keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
        if keys:
            return keys

    logger.critical(
        "CARQ_API_KEYS is not set or empty. "
        "All API requests will be rejected until this is configured. "
        "Set CARQ_API_KEYS to a comma-separated list of strong random API keys."
    )
    return frozenset()


class APIKeyAuth:
    """Handler de autenticação por chave de API (variante Bearer token)."""

    def __init__(self, valid_keys: frozenset[str]):
        self.valid_keys = valid_keys
        self.logger = logging.getLogger(__name__)

    async def __call__(
        self,
        authorization: Optional[str] = Header(None),
    ) -> str:
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization format (use: Bearer {api_key})",
            )

        api_key = parts[1]
        if api_key not in self.valid_keys:
            self.logger.warning("Invalid API key attempt (bearer)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        return api_key


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
) -> str:
    """Verifica o cabeçalho X-API-Key contra o conjunto de chaves configurado.

    Raises:
        HTTPException 401: Chave ausente ou não está no conjunto configurado.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    valid_keys = _load_valid_keys()

    # Nenhuma chave configurada — bloqueia tudo
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key store not configured. Contact the administrator.",
        )

    if x_api_key not in valid_keys:
        logger.warning("Rejected request with unknown API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key
