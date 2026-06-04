"""Base declarativa do SQLAlchemy e mixins de timestamps/UUID para todos os modelos ORM."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, declarative_base, mapped_column
from sqlalchemy.types import TypeDecorator


class DialectJSON(TypeDecorator):
    """Tipo JSON usando JSONB no PostgreSQL e JSON em outros bancos de dados (ex.: SQLite para testes)."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class BaseModel:
    """Mixin base para todos os modelos ORM com timestamps e UUID."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        columns = [col.name for col in self.__table__.columns if not col.name.startswith("_")]
        values = ", ".join(f"{col}={getattr(self, col, None)!r}" for col in columns[:3])
        return f"<{self.__class__.__name__}({values})>"

    def to_dict(self) -> dict[str, Any]:
        """Converte a instância do modelo em dicionário."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, (datetime, uuid.UUID)):
                value = str(value)
            elif hasattr(value, "to_dict"):
                value = value.to_dict()
            result[column.name] = value
        return result


Base = declarative_base(cls=BaseModel)
