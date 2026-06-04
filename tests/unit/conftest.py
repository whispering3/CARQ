"""
Conftest for unit tests.
Provides unit-test-specific fixtures and configuration.
"""

from unittest.mock import AsyncMock, Mock

import pytest

# ============================================================================
# MOCK FIXTURES FOR EXTERNAL SERVICES
# ============================================================================


@pytest.fixture
def mock_openai_embedding_response():
    """Create mock OpenAI embedding response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1] * 1536,
            },
            {
                "object": "embedding",
                "index": 1,
                "embedding": [0.2] * 1536,
            },
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
        },
    }


@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_client.post = AsyncMock()
    mock_client.put = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    mock = Mock()
    mock.debug = Mock()
    mock.info = Mock()
    mock.warning = Mock()
    mock.error = Mock()
    mock.critical = Mock()
    return mock


# ============================================================================
# JWT TOKEN FIXTURES
# ============================================================================


@pytest.fixture
def valid_jwt_token():
    """Create a valid JWT token for testing."""
    from datetime import datetime, timedelta

    import jwt

    secret = "test-secret-key"
    payload = {
        "sub": "test-user",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token


@pytest.fixture
def expired_jwt_token():
    """Create an expired JWT token for testing."""
    from datetime import datetime, timedelta

    import jwt

    secret = "test-secret-key"
    payload = {
        "sub": "test-user",
        "iat": datetime.utcnow() - timedelta(hours=2),
        "exp": datetime.utcnow() - timedelta(hours=1),
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token


@pytest.fixture
def invalid_jwt_token():
    """Create an invalid JWT token for testing."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.invalid"


# ============================================================================
# API FIXTURES
# ============================================================================


@pytest.fixture
def sample_ingest_request():
    """Create sample ingest request."""
    return {
        "file_path": "/tmp/test.pdf",
        "metadata": {
            "category": "test",
            "source": "unit_test",
        },
        "priority": 5,
    }


@pytest.fixture
def sample_search_request():
    """Create sample search request."""
    return {
        "query": "machine learning algorithms",
        "limit": 10,
        "similarity_threshold": 0.7,
        "model": "text-embedding-3-small",
    }


# ============================================================================
# PDF TESTING FIXTURES
# ============================================================================


@pytest.fixture
def sample_pdf_bytes():
    """Create sample PDF bytes for testing."""
    try:
        from io import BytesIO

        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)

        # Page 1
        c.drawString(100, 750, "Test PDF Document")
        c.drawString(100, 730, "This is a test document for PDF parsing.")
        c.drawString(100, 710, "It contains multiple lines of text.")

        c.showPage()

        # Page 2
        c.drawString(100, 750, "Page 2 - Additional Content")
        c.drawString(100, 730, "This is the second page of the test document.")
        c.drawString(100, 710, "It has more content for testing semantic chunking.")

        c.showPage()

        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except ImportError:
        # Fallback if reportlab is not available
        return b"%PDF-1.4\n%Test PDF content"


@pytest.fixture
def corrupted_pdf_bytes():
    """Create corrupted PDF bytes for testing."""
    return b"This is not a valid PDF file content"


@pytest.fixture
def empty_pdf_bytes():
    """Create empty PDF bytes for testing."""
    return b""


# ============================================================================
# CHUNK TESTING FIXTURES
# ============================================================================


@pytest.fixture
def sample_text_content():
    """Create sample text content for chunking tests."""
    return """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses on
    developing algorithms that can learn from and make decisions based on data.

    Chapter 1: Fundamentals

    The fundamentals of machine learning include understanding the basic concepts
    of supervised learning, unsupervised learning, and reinforcement learning.

    Supervised learning involves training models with labeled data where each
    input has an associated output. Common algorithms include decision trees,
    random forests, and neural networks.

    Unsupervised learning works with unlabeled data to discover patterns and
    structures within the data. Clustering and dimensionality reduction are
    common techniques.

    Reinforcement learning involves training agents to make sequential decisions
    in environments to maximize cumulative rewards.

    Chapter 2: Advanced Topics

    More advanced topics include deep learning, transfer learning, and
    meta-learning. These techniques push the boundaries of what is possible
    with machine learning.
    """


@pytest.fixture
def short_text_content():
    """Create short text content."""
    return "This is a short text content for testing."


@pytest.fixture
def long_text_content():
    """Create long text content."""
    paragraph = """
    This is a paragraph about machine learning. Machine learning is the field
    of artificial intelligence that enables computers to learn from data without
    being explicitly programmed. It has applications in image recognition,
    natural language processing, recommendation systems, and many more areas.
    """ * 50
    return paragraph


# ============================================================================
# EMBEDDING FIXTURES
# ============================================================================


@pytest.fixture
def sample_embedding_vector():
    """Create sample embedding vector."""
    return [0.1] * 1536  # OpenAI embedding dimension


@pytest.fixture
def multiple_embedding_vectors():
    """Create multiple embedding vectors."""
    return [
        [0.1 + i * 0.001] * 1536 for i in range(10)
    ]


# ============================================================================
# DATABASE FIXTURES
# ============================================================================


@pytest.fixture
def sample_postgres_url():
    """Create sample PostgreSQL URL."""
    return "postgresql+psycopg://user:password@localhost:5432/test_db"


@pytest.fixture
def sample_sqlite_url():
    """Create sample SQLite URL."""
    return "sqlite+aiosqlite:///:memory:"


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def sample_env_vars(monkeypatch):
    """Set sample environment variables."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key")
    monkeypatch.setenv("CARQ_DB_HOST", "localhost")
    monkeypatch.setenv("CARQ_DB_PORT", "5432")
    monkeypatch.setenv("CARQ_DB_USER", "postgres")
    monkeypatch.setenv("CARQ_DB_PASSWORD", "postgres")
    monkeypatch.setenv("CARQ_DB_DATABASE", "carq_test")
    monkeypatch.setenv("CARQ_REDIS_HOST", "localhost")
    monkeypatch.setenv("CARQ_REDIS_PORT", "6379")
    return monkeypatch


# ============================================================================
# PYTEST MARKERS
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def register_unit_markers():
    """Register unit test markers."""
    pass
