"""Análise assíncrona de PDFs com estratégias de fallback para arquivos corrompidos."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import pypdf

from carq.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)


class PDFParsingError(ProcessingError):
    """Disparado quando a análise de PDF falha."""

    pass


@dataclass
class PDFMetadata:
    """Metadados extraídos do PDF."""

    filename: str
    num_pages: int
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    content_hash: str = ""
    file_size_bytes: int = 0
    encryption_status: str = "none"


@dataclass
class ParsedPage:
    """Resultado de extração de uma única página."""

    page_num: int
    text: str
    images_found: int = 0
    extraction_method: str = "pypdf"
    success: bool = True
    error: Optional[str] = None


class PDFParser:
    """Analisador assíncrono de PDF com estratégias de fallback."""

    def __init__(self, max_workers: int = 4, chunk_size: int = 8192):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    async def parse(
        self,
        file_path: Path,
        extract_metadata: bool = True,
        skip_corrupted: bool = False,
    ) -> tuple[list[ParsedPage], PDFMetadata]:
        """Analisa arquivo PDF de forma assíncrona."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            file_content = await self._read_file_async(file_path)
            file_hash = hashlib.sha256(file_content).hexdigest()

            pdf_file = pypdf.PdfReader(BytesIO(file_content))

            metadata = self._extract_metadata(
                pdf_file,
                file_path,
                file_content,
                file_hash,
            )

            pages = await self._extract_pages(pdf_file, skip_corrupted)

            self.logger.info(
                f"Parsed PDF {file_path.name}",
                extra={
                    "pdf_filename": file_path.name,
                    "pages": len(pages),
                    "content_hash": file_hash[:16],
                },
            )

            return pages, metadata

        except pypdf.errors.PdfReadError as e:
            self.logger.error(f"PDF read error: {e}", extra={"file": str(file_path)})
            raise PDFParsingError(f"Invalid PDF format: {e}") from e
        except Exception as e:
            self.logger.error(
                f"Unexpected parsing error: {e}",
                extra={"file": str(file_path), "error_type": type(e).__name__},
            )
            raise PDFParsingError(f"Failed to parse PDF: {e}") from e

    async def _read_file_async(self, file_path: Path) -> bytes:
        """Lê o arquivo de forma assíncrona em blocos."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: file_path.read_bytes(),
        )

    def _extract_metadata(
        self,
        pdf: pypdf.PdfReader,
        file_path: Path,
        file_content: bytes,
        content_hash: str,
    ) -> PDFMetadata:
        """Extrai metadados do PDF."""
        metadata_dict = pdf.metadata or {}

        return PDFMetadata(
            filename=file_path.name,
            num_pages=len(pdf.pages),
            title=metadata_dict.get("/Title", "").strip() or None,
            author=metadata_dict.get("/Author", "").strip() or None,
            creation_date=metadata_dict.get("/CreationDate", "").strip() or None,
            content_hash=content_hash,
            file_size_bytes=len(file_content),
            encryption_status="encrypted" if pdf.is_encrypted else "none",
        )

    async def _extract_pages(
        self,
        pdf: pypdf.PdfReader,
        skip_corrupted: bool = False,
    ) -> list[ParsedPage]:
        """Extrai texto de todas as páginas de forma assíncrona."""
        loop = asyncio.get_event_loop()
        tasks = []

        for page_num, page in enumerate(pdf.pages):
            task = loop.run_in_executor(
                None,
                lambda p=page, n=page_num: self._extract_single_page(p, n),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if skip_corrupted:
                    self.logger.warning(
                        f"Skipping corrupted page {i}",
                        extra={"error": str(result)},
                    )
                    continue
                # Ainda adiciona a página com marcador de erro
                pages.append(
                    ParsedPage(
                        page_num=i,
                        text="",
                        success=False,
                        error=str(result),
                    )
                )
            else:
                pages.append(result)

        return pages

    def _extract_single_page(
        self,
        page: pypdf.PageObject,
        page_num: int,
    ) -> ParsedPage:
        """Extrai texto de uma única página."""
        try:
            text = page.extract_text()
            # Trata extrações vazias (páginas apenas com imagens)
            if not text or not text.strip():
                text = f"[Image-only page {page_num + 1}]"

            return ParsedPage(
                page_num=page_num,
                text=text,
                extraction_method="pypdf",
                success=True,
            )
        except Exception as e:
            self.logger.debug(
                f"Page extraction error {page_num}: {e}",
                extra={"page": page_num, "error": str(e)},
            )
            return ParsedPage(
                page_num=page_num,
                text="",
                success=False,
                error=str(e),
            )

    async def parse_batch(
        self,
        file_paths: list[Path],
        skip_corrupted: bool = False,
    ) -> list[tuple[Path, list[ParsedPage], PDFMetadata]]:
        """Analisa múltiplos PDFs de forma concorrente."""
        tasks = [
            self.parse(file_path, extract_metadata=True, skip_corrupted=skip_corrupted)
            for file_path in file_paths
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for file_path, result in zip(file_paths, results, strict=False):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Batch parse failed for {file_path.name}",
                    extra={"error": str(result)},
                )
                continue
            pages, metadata = result
            output.append((file_path, pages, metadata))

        return output

    @staticmethod
    def calculate_content_hash(content: bytes) -> str:
        """Calcula o hash SHA-256 do conteúdo do PDF."""
        return hashlib.sha256(content).hexdigest()
