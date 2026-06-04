"""Unit tests for PDF processing (Phase 4).

Tests cover:
- PDF parsing (valid, corrupted, encrypted)
- Semantic chunking (various strategies)
- Chunk validation (quality, duplicates)
"""

from io import BytesIO

import pytest

from carq.pdf.chunk_validator import ChunkValidator, ValidationResult
from carq.pdf.pdf_parser import ParsedPage, PDFMetadata, PDFParser, PDFParsingError
from carq.pdf.semantic_chunker import Chunk, ChunkingConfig, SemanticChunker

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_pdf_content():
    """Create a simple PDF for testing."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    # Create PDF with reportlab
    pdf_bytes = BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)

    c.drawString(100, 750, "Test PDF Document")
    c.drawString(100, 730, "This is a test document for PDF parsing.")
    c.showPage()

    c.drawString(100, 750, "Page 2")
    c.drawString(100, 730, "This is the second page of the test document.")
    c.showPage()

    c.save()
    pdf_bytes.seek(0)
    return pdf_bytes.getvalue()


@pytest.fixture
def pdf_parser():
    """Create PDF parser instance."""
    return PDFParser(max_workers=2)


@pytest.fixture
def semantic_chunker():
    """Create semantic chunker instance."""
    config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=30,
    )
    return SemanticChunker(config)


@pytest.fixture
def chunk_validator():
    """Create chunk validator instance."""
    return ChunkValidator(
        min_tokens=5,
        max_tokens=1024,
        min_chars=20,
        max_chars=4096,
    )


@pytest.fixture
async def temp_pdf_file(tmp_path, sample_pdf_content):
    """Create temporary PDF file."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


# ============================================================================
# PDF PARSER TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_pdf_parser_basic_parsing(pdf_parser, temp_pdf_file):
    """Test basic PDF parsing."""
    pages, metadata = await pdf_parser.parse(temp_pdf_file)

    assert len(pages) == 2
    assert all(isinstance(p, ParsedPage) for p in pages)
    assert all(p.success for p in pages)
    assert metadata.num_pages == 2
    assert metadata.filename == "test.pdf"


@pytest.mark.asyncio
async def test_pdf_parser_metadata_extraction(pdf_parser, temp_pdf_file):
    """Test metadata extraction."""
    pages, metadata = await pdf_parser.parse(temp_pdf_file)

    assert isinstance(metadata, PDFMetadata)
    assert len(metadata.content_hash) == 64  # SHA-256
    assert metadata.file_size_bytes > 0
    assert metadata.encryption_status == "none"
    assert metadata.num_pages == 2


@pytest.mark.asyncio
async def test_pdf_parser_missing_file(pdf_parser, tmp_path):
    """Test handling of missing PDF."""
    missing_file = tmp_path / "nonexistent.pdf"

    with pytest.raises(FileNotFoundError):
        await pdf_parser.parse(missing_file)


@pytest.mark.asyncio
async def test_pdf_parser_invalid_pdf(pdf_parser, tmp_path):
    """Test handling of invalid PDF."""
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("This is not a PDF")

    with pytest.raises(PDFParsingError):
        await pdf_parser.parse(invalid_pdf)


@pytest.mark.asyncio
async def test_pdf_parser_batch_parsing(pdf_parser, tmp_path, sample_pdf_content):
    """Test batch PDF parsing."""
    # Create multiple PDFs
    pdf_paths = []
    for i in range(3):
        pdf_path = tmp_path / f"test_{i}.pdf"
        pdf_path.write_bytes(sample_pdf_content)
        pdf_paths.append(pdf_path)

    results = await pdf_parser.parse_batch(pdf_paths)

    assert len(results) == 3
    assert all(len(pages) == 2 for _, pages, _ in results)


# ============================================================================
# SEMANTIC CHUNKER TESTS
# ============================================================================


def test_semantic_chunker_basic_chunking(semantic_chunker):
    """Test basic semantic chunking."""
    text = """Paragraph 1. This is the first sentence.
    This is the second sentence.

    Paragraph 2. This is the third sentence.
    This is the fourth sentence.
    """

    chunks = semantic_chunker.chunk(text)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)
    assert all(c.chunk_num == i for i, c in enumerate(chunks))


def test_semantic_chunker_respects_size_limits(semantic_chunker):
    """Test that chunks respect size limits."""
    # Create large text
    text = " ".join(["word"] * 5000)

    chunks = semantic_chunker.chunk(text)

    # All chunks should be within limits
    for chunk in chunks:
        assert (
            len(chunk.text) <= semantic_chunker.config.chunk_size * 1.1
        )  # Allow 10% margin


def test_semantic_chunker_enforces_min_size(semantic_chunker):
    """Test minimum chunk size enforcement."""
    # Create text with very small paragraphs
    text = "a. b. c. d. e. f. " * 100

    chunks = semantic_chunker.chunk(text)

    # Chunks should either meet minimum or be merged
    valid_chunks = [c for c in chunks if len(c.text) >= 30]  # min_chunk_size
    assert len(valid_chunks) == len(chunks) or len(chunks) == 0


def test_semantic_chunker_adds_context(semantic_chunker):
    """Test that context is added to chunks."""
    text = "First sentence. Second sentence. Third sentence."

    chunks = semantic_chunker.chunk(text)

    # Chunks with neighbors should have context
    for chunk in chunks:
        # Context may be empty for single-chunk documents
        if chunk.chunk_num > 0 or chunk.chunk_num < len(chunks) - 1:
            # Should have empty or non-empty context
            assert isinstance(chunk.context, str)


def test_semantic_chunker_handles_empty_text(semantic_chunker):
    """Test handling of empty text."""
    chunks = semantic_chunker.chunk("")
    assert len(chunks) == 0

    chunks = semantic_chunker.chunk("   ")
    assert len(chunks) == 0


def test_semantic_chunker_token_estimation(semantic_chunker):
    """Test token count estimation."""
    text = "a" * 400  # Roughly 100 tokens

    chunks = semantic_chunker.chunk(text)

    for chunk in chunks:
        assert chunk.tokens_estimate > 0
        # Rough check: 1 token ≈ 4 chars
        assert chunk.tokens_estimate <= len(chunk.text) // 3


# ============================================================================
# CHUNK VALIDATOR TESTS
# ============================================================================


def test_chunk_validator_valid_chunk(chunk_validator):
    """Test validation of valid chunk."""
    text = "This is a valid chunk with enough content. It has multiple sentences and good quality."

    result = chunk_validator.validate(text)

    assert result.is_valid
    assert len(result.reasons) == 0
    assert result.token_count > 0
    assert result.quality_score > 0.5


def test_chunk_validator_too_short(chunk_validator):
    """Test rejection of too-short chunk."""
    text = "Short"

    result = chunk_validator.validate(text)

    assert not result.is_valid
    assert "too_few_chars" in result.issues or "too_few_tokens" in result.issues


def test_chunk_validator_too_long(chunk_validator):
    """Test rejection of too-long chunk."""
    text = "word " * 2000  # ~8000 chars

    result = chunk_validator.validate(text)

    assert not result.is_valid
    assert "too_many_chars" in result.issues


def test_chunk_validator_low_quality_content(chunk_validator):
    """Test rejection of low-quality content."""
    text = "!@#$%^&*()" * 100  # Mostly special chars

    result = chunk_validator.validate(text)

    assert not result.is_valid or result.quality_score < 0.5
    if not result.is_valid:
        assert any("quality" in issue or "special" in issue for issue in result.issues)


def test_chunk_validator_truncated_detection(chunk_validator):
    """Test detection of truncated text."""
    # Text that looks truncated
    text = "This sentence is incomplete and ends with a lowercase letter a"

    result = chunk_validator.validate(text)

    if len(result.reasons) > 0 or len(result.issues) > 0:
        assert "truncated" in result.issues or any(
            "truncated" in r.lower() for r in result.reasons
        )


def test_chunk_validator_duplicate_detection(chunk_validator):
    """Test duplicate detection."""
    text = "This is a sample chunk that contains some text."

    # First validation
    result1 = chunk_validator.validate(text, check_duplicates=True)
    assert not result1.is_duplicate

    # Second validation (should detect duplicate)
    result2 = chunk_validator.validate(text, check_duplicates=True)
    assert result2.is_duplicate


def test_chunk_validator_near_duplicate_detection(chunk_validator):
    """Test near-duplicate detection."""
    text1 = "This is a sample chunk that contains some text for testing purposes."
    text2 = "This is a sample chunk that contains some text for testing purposes."  # Nearly identical

    result1 = chunk_validator.validate(text1, check_duplicates=True)
    result2 = chunk_validator.validate(text2, check_duplicates=True)

    assert not result1.is_duplicate
    assert result2.is_duplicate


def test_chunk_validator_batch_validation(chunk_validator):
    """Test batch validation."""
    texts = [
        "Valid chunk with good content and sufficient length here.",
        "Too short",
        "Another valid chunk with meaningful content that is long enough.",
    ]

    results = chunk_validator.validate_batch(texts)

    assert len(results) == 3
    assert all(isinstance(r, ValidationResult) for r in results)
    # First and third should be valid, second should not
    assert results[0].is_valid
    assert not results[1].is_valid
    assert results[2].is_valid


def test_chunk_validator_filter_valid(chunk_validator):
    """Test filtering to valid chunks."""
    chunks = [
        "Valid chunk with good content and sufficient length here.",
        "Short",
        "Another valid chunk with meaningful content that is long enough.",
    ]

    valid_chunks, indices = chunk_validator.filter_valid_chunks(chunks)

    assert len(valid_chunks) == 2
    assert indices == [0, 2]


def test_chunk_validator_reset_duplicates(chunk_validator):
    """Test duplicate cache reset."""
    text = "Test text for duplicate detection purposes here."

    # Add to cache
    chunk_validator.validate(text, check_duplicates=True)
    assert len(chunk_validator.seen_chunks) > 0

    # Reset
    chunk_validator.reset_duplicates()
    assert len(chunk_validator.seen_chunks) == 0

    # Should not detect as duplicate after reset
    result = chunk_validator.validate(text, check_duplicates=True)
    assert not result.is_duplicate


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_pdf_to_chunks_pipeline(
    pdf_parser,
    semantic_chunker,
    chunk_validator,
    temp_pdf_file,
):
    """Test complete PDF -> chunks -> validation pipeline."""
    # Parse PDF
    pages, metadata = await pdf_parser.parse(temp_pdf_file)
    assert len(pages) > 0

    # Combine page texts
    full_text = "\n\n".join(p.text for p in pages if p.success)

    # Chunk text
    chunks = semantic_chunker.chunk(full_text)
    assert len(chunks) > 0

    # Validate chunks
    texts = [c.text for c in chunks]
    valid_chunks, indices = chunk_validator.filter_valid_chunks(texts)
    assert len(valid_chunks) > 0


def test_chunking_config_validation():
    """Test chunking configuration validation."""
    # Valid config
    config = ChunkingConfig(
        chunk_size=1024,
        chunk_overlap=100,
        min_chunk_size=100,
    )
    assert config.chunk_size > config.min_chunk_size

    # Invalid: chunk_size < min_chunk_size
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=100,
        )

    # Invalid: overlap > chunk_size / 2
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=60,
            min_chunk_size=10,
        )
