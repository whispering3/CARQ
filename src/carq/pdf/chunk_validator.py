"""Validação de qualidade de chunks para embedding e recuperação."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado da validação de um chunk."""

    is_valid: bool
    reasons: list[str]
    token_count: int = 0
    quality_score: float = 1.0
    is_duplicate: bool = False
    issues: list[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class ChunkValidator:
    """Valida a qualidade dos chunks e detecta duplicatas."""

    def __init__(
        self,
        min_tokens: int = 10,
        max_tokens: int = 2048,
        min_chars: int = 50,
        max_chars: int = 8192,
        require_alphanumeric_ratio: float = 0.5,
        duplicate_similarity_threshold: float = 0.95,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.require_alphanumeric_ratio = require_alphanumeric_ratio
        self.duplicate_similarity_threshold = duplicate_similarity_threshold

        self.seen_chunks: dict[str, str] = {}  # hash -> text
        self.logger = logging.getLogger(__name__)

    def validate(
        self,
        text: str,
        check_duplicates: bool = True,
    ) -> ValidationResult:
        """Valida um único chunk."""
        reasons = []
        issues = []

        text_clean = text.strip()

        token_count = self._estimate_tokens(text_clean)
        if token_count < self.min_tokens:
            reasons.append(f"Token count {token_count} < minimum {self.min_tokens}")
            issues.append("too_few_tokens")
        if token_count > self.max_tokens:
            reasons.append(
                f"Token count {token_count} > maximum {self.max_tokens}"
            )
            issues.append("too_many_tokens")

        char_count = len(text_clean)
        if char_count < self.min_chars:
            reasons.append(
                f"Character count {char_count} < minimum {self.min_chars}"
            )
            issues.append("too_few_chars")
        if char_count > self.max_chars:
            reasons.append(
                f"Character count {char_count} > maximum {self.max_chars}"
            )
            issues.append("too_many_chars")

        if not self._has_meaningful_content(text_clean):
            reasons.append("Insufficient meaningful content (mostly special chars)")
            issues.append("low_content_quality")

        if self._is_likely_truncated(text_clean):
            reasons.append("Text appears to be truncated")
            issues.append("truncated_content")

        is_duplicate = False
        if check_duplicates:
            is_duplicate = self._is_duplicate(text_clean)
            if is_duplicate:
                reasons.append("Duplicate or near-duplicate content")
                issues.append("duplicate")

        quality_score = self._calculate_quality_score(
            text_clean,
            token_count,
            is_duplicate,
        )

        is_valid = len(reasons) == 0

        return ValidationResult(
            is_valid=is_valid,
            reasons=reasons,
            token_count=token_count,
            quality_score=quality_score,
            is_duplicate=is_duplicate,
            issues=issues,
        )

    def validate_batch(
        self,
        texts: list[str],
        check_duplicates: bool = True,
    ) -> list[ValidationResult]:
        """Valida múltiplos chunks."""
        return [
            self.validate(text, check_duplicates=check_duplicates) for text in texts
        ]

    def filter_valid_chunks(
        self,
        chunks: list[str],
        check_duplicates: bool = True,
    ) -> tuple[list[str], list[int]]:
        """Filtra para manter apenas chunks válidos. Retorna (chunks_válidos, índices_válidos)."""
        valid_chunks = []
        valid_indices = []

        for i, chunk in enumerate(chunks):
            result = self.validate(chunk, check_duplicates=check_duplicates)
            if result.is_valid:
                valid_chunks.append(chunk)
                valid_indices.append(i)

        return valid_chunks, valid_indices

    def reset_duplicates(self):
        self.seen_chunks.clear()
        self.logger.debug("Duplicate cache reset")

    def _estimate_tokens(self, text: str) -> int:
        """Estima a contagem de tokens usando proporção de caracteres.

        Aproximação simples: 1 token ≈ 4 caracteres.
        Mais preciso que len(text.split()) para tokens de subpalavra.
        """
        return max(1, len(text) // 4)

    def _has_meaningful_content(self, text: str) -> bool:
        """Verifica se o texto possui conteúdo significativo suficiente.

        Conta alfanuméricos + pontuação comum.
        """
        alphanumeric = sum(
            1 for c in text if c.isalnum() or c in " .,!?;:-\n\t"
        )
        total = len(text)

        if total == 0:
            return False

        ratio = alphanumeric / total
        return ratio >= self.require_alphanumeric_ratio

    def _is_likely_truncated(self, text: str) -> bool:
        """Detecta se o texto parece truncado (no meio de uma sentença).

        Procura pontuação terminal ao final da última sentença, em vez de
        verificar o último caractere, o que gerava muitos falsos positivos para
        chunks válidos terminados em abreviações, números ou expressões entre parênteses.

        Heurísticas:
        - Sem pontuação terminal (.  !  ?  :  …  —  ") ao final da sentença
        - Termina com hífen no meio de palavra (artefato de quebra de linha)
        """
        if not text:
            return False

        last_char = text[-1]

        # Hífen explícito no meio de palavra indica sempre truncamento
        if last_char == "-":
            return True

        # Termina com pontuação terminal reconhecida → não truncado
        if last_char in ".!?:…\")]}—":
            return False

        # Qualquer outra coisa (letra minúscula, maiúscula, dígito, etc.): verifica
        # se a última *sentença* (dividida por fronteiras de sentença) possui uma
        # marcação terminal. Isso evita falsos positivos para chunks que terminam
        # com, por exemplo, um substantivo próprio, um acrônimo ou um ano.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        last_sentence = sentences[-1].strip() if sentences else ""
        if last_sentence and last_sentence[-1] in ".!?:…":
            return False

        # Nenhuma marcação terminal encontrada — trata como truncado
        return True

    def _is_duplicate(self, text: str) -> bool:
        """Verifica se o chunk é duplicado ou quase duplicado.

        Correspondências exatas são detectadas em O(1) via um conjunto de hashes SHA-256.
        A detecção de quase duplicatas (similaridade Jaccard) é aplicada apenas quando
        nenhuma correspondência exata é encontrada, mantendo o custo médio baixo.
        """
        import hashlib

        text_clean = text.lower().strip()
        text_hash = hashlib.sha256(text_clean.encode()).hexdigest()

        if text_hash in self.seen_chunks:
            return True

        for seen_text in self.seen_chunks.values():
            if self._jaccard_similarity(text_clean, seen_text) > (
                self.duplicate_similarity_threshold
            ):
                return True

        self.seen_chunks[text_hash] = text_clean
        return False

    @staticmethod
    def _text_hash(text: str) -> str:
        """Hash SHA-256 determinístico do texto (independente de PYTHONHASHSEED)."""
        import hashlib

        return hashlib.sha256(text.strip().encode()).hexdigest()

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Calcula a similaridade de Jaccard entre dois textos.

        Usa tokens de palavras, normalizado para [0, 1].
        """
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _calculate_quality_score(
        self,
        text: str,
        token_count: int,
        is_duplicate: bool,
    ) -> float:
        """Pontuação de qualidade de 0.0 a 1.0 com penalidades por tokens, conteúdo e duplicata."""
        score = 1.0

        ideal_min = max(self.min_tokens * 2, 10)
        ideal_max = min(self.max_tokens, 1024)

        if token_count < ideal_min:
            score *= token_count / ideal_min
        elif token_count > ideal_max:
            score *= ideal_max / token_count

        alphanumeric = sum(
            1 for c in text if c.isalnum() or c in " .,!?;:-\n\t"
        )
        if len(text) > 0:
            quality_ratio = alphanumeric / len(text)
            if quality_ratio < 0.7:
                score *= quality_ratio / 0.7

        if is_duplicate:
            score *= 0.3

        return max(0.0, min(1.0, score))
