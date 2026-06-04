"""Segmentação semântica de documentos respeitando fronteiras de parágrafo e sentença."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuração para segmentação semântica."""

    chunk_size: int = 1024
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    sentence_split_pattern: str = r"(?<=[.!?])\s+"
    paragraph_split_pattern: str = r"\n\n+"
    preserve_headers: bool = True
    max_chunks_per_doc: int = 10000

    def __post_init__(self):
        if self.chunk_size < self.min_chunk_size:
            raise ValueError("chunk_size must be >= min_chunk_size")
        if self.chunk_overlap > self.chunk_size // 2:
            raise ValueError("chunk_overlap must be <= chunk_size / 2")


@dataclass
class Chunk:
    """Chunk extraído com metadados de posição e contexto."""

    text: str
    chunk_num: int
    start_char: int = 0
    end_char: int = 0
    page_num: Optional[int] = None
    context: str = ""
    tokens_estimate: int = 0

    def __post_init__(self):
        # Estimativa aproximada: 1 token ≈ 4 caracteres
        self.tokens_estimate = max(1, len(self.text) // 4)


class SemanticChunker:
    """Segmentação inteligente de documentos respeitando fronteiras semânticas."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.logger = logging.getLogger(__name__)

    def chunk(
        self,
        text: str,
        page_num: Optional[int] = None,
    ) -> list[Chunk]:
        """Segmenta o texto em chunks semânticos."""
        if not text or not text.strip():
            self.logger.warning("Empty text provided to chunker")
            return []

        text = self._clean_text(text)

        chunks = self._split_semantic(text)
        chunks = self._enforce_chunk_size(chunks)
        chunks = self._add_context(chunks, text)

        for i, chunk in enumerate(chunks):
            chunk.chunk_num = i
            if page_num is not None:
                chunk.page_num = page_num

        if len(chunks) > self.config.max_chunks_per_doc:
            self.logger.warning(
                f"Document exceeds max chunks: {len(chunks)} > "
                f"{self.config.max_chunks_per_doc}",
                extra={"chunk_count": len(chunks)},
            )
            chunks = chunks[: self.config.max_chunks_per_doc]

        self.logger.debug(
            f"Generated {len(chunks)} chunks",
            extra={
                "chunk_count": len(chunks),
                "avg_size": sum(len(c.text) for c in chunks) // len(chunks)
                if chunks
                else 0,
            },
        )

        return chunks

    def chunk_batch(
        self,
        documents: dict[int, str],
    ) -> dict[int, list[Chunk]]:
        """Segmenta múltiplos documentos. Recebe mapeamento de doc_id→texto."""
        results = {}
        for doc_id, text in documents.items():
            results[doc_id] = self.chunk(text, page_num=doc_id)
        return results

    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto preservando fronteiras de parágrafo.

        Normaliza espaços em branco inline (espaços/tabs) por parágrafo, mas mantém
        os separadores de parágrafo com dupla quebra de linha intactos para que _split_semantic()
        possa usá-los para divisão semântica.
        """
        # Divide nas fronteiras de parágrafo primeiro, depois normaliza cada parágrafo
        # individualmente — isso preserva os separadores \\n\\n que _split_semantic
        # usa para detectar quebras de parágrafo.
        paragraphs = re.split(r"\n\n+", text)
        cleaned = []
        for para in paragraphs:
            # Colapsa espaços/tabs inline; remove caracteres de controle exceto quebras de linha
            para = re.sub(r"[ \t]+", " ", para)
            para = "".join(c for c in para if c.isprintable() or c == "\n")
            para = para.strip()
            if para:
                cleaned.append(para)
        return "\n\n".join(cleaned)

    def _split_semantic(self, text: str) -> list[Chunk]:
        """Divide o texto respeitando fronteiras semânticas."""
        paragraphs = re.split(self.config.paragraph_split_pattern, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""
        start_pos = 0

        for para in paragraphs:
            test_chunk = f"{current_chunk}\n\n{para}".strip()

            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    end_pos = start_pos + len(current_chunk)
                    chunks.append(
                        Chunk(
                            text=current_chunk,
                            chunk_num=len(chunks),
                            start_char=start_pos,
                            end_char=end_pos,
                        )
                    )
                    start_pos = end_pos

                sentences = re.split(
                    self.config.sentence_split_pattern,
                    para,
                )
                sentences = [s.strip() for s in sentences if s.strip()]

                current_chunk = ""
                for sentence in sentences:
                    test_chunk = f"{current_chunk} {sentence}".strip()
                    if len(test_chunk) <= self.config.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            end_pos = start_pos + len(current_chunk)
                            chunks.append(
                                Chunk(
                                    text=current_chunk,
                                    chunk_num=len(chunks),
                                    start_char=start_pos,
                                    end_char=end_pos,
                                )
                            )
                            start_pos = end_pos
                        current_chunk = sentence

        if current_chunk:
            end_pos = start_pos + len(current_chunk)
            chunks.append(
                Chunk(
                    text=current_chunk,
                    chunk_num=len(chunks),
                    start_char=start_pos,
                    end_char=end_pos,
                )
            )

        return chunks

    def _enforce_chunk_size(self, chunks: list[Chunk]) -> list[Chunk]:
        """Aplica tamanhos mínimo e máximo de chunk."""
        result = []

        for chunk in chunks:
            if len(chunk.text) < self.config.min_chunk_size:
                if result and len(result[-1].text) + len(chunk.text) < (
                    self.config.chunk_size
                ):
                    result[-1].text = (
                        f"{result[-1].text}\n\n{chunk.text}"
                    )
                    result[-1].end_char = chunk.end_char
                continue

            if len(chunk.text) > self.config.chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                result.extend(sub_chunks)
            else:
                result.append(chunk)

        return result

    def _split_large_chunk(self, chunk: Chunk) -> list[Chunk]:
        """Divide chunk muito grande por tamanho fixo."""
        text = chunk.text
        chunks = []
        start_pos = chunk.start_char

        while len(text) > self.config.chunk_size:
            split_point = self.config.chunk_size
            last_space = text.rfind(" ", 0, split_point)
            if last_space > self.config.chunk_size * 0.7:
                split_point = last_space

            sub_text = text[:split_point].strip()
            if sub_text:
                chunks.append(
                    Chunk(
                        text=sub_text,
                        chunk_num=len(chunks),
                        start_char=start_pos,
                        end_char=start_pos + len(sub_text),
                    )
                )
                start_pos += len(sub_text) + 1

            text = text[split_point:].strip()

        if text:
            chunks.append(
                Chunk(
                    text=text,
                    chunk_num=len(chunks),
                    start_char=start_pos,
                    end_char=chunk.end_char,
                )
            )

        return chunks

    def _add_context(self, chunks: list[Chunk], full_text: str) -> list[Chunk]:
        """Adiciona contexto circundante aos chunks para embeddings."""
        for i, chunk in enumerate(chunks):
            context_parts = []

            if i > 0:
                prev_chunk = chunks[i - 1]
                context_parts.append(f"...{prev_chunk.text[-100:]}")

            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                context_parts.append(f"{next_chunk.text[:100]}...")

            chunk.context = " ".join(context_parts) if context_parts else ""

        return chunks
