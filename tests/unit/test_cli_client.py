"""
Unit tests for CLI client (carq.cli.client).
Uses Click's CliRunner to invoke commands without a real HTTP server.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from carq.cli.client import cli, make_request, get_headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int = 200, json_data: dict = None):
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Tests: get_headers
# ---------------------------------------------------------------------------

class TestGetHeaders:
    def test_no_token(self, monkeypatch):
        monkeypatch.setenv("CARQ_API_TOKEN", "")
        import importlib, carq.cli.client as m
        importlib.reload(m)
        headers = m.get_headers()
        assert "Content-Type" in headers
        assert "Authorization" not in headers

    def test_with_token(self, monkeypatch):
        monkeypatch.setenv("CARQ_API_TOKEN", "mytoken")
        import importlib, carq.cli.client as m
        importlib.reload(m)
        headers = m.get_headers()
        assert headers.get("Authorization") == "Bearer mytoken"


# ---------------------------------------------------------------------------
# Tests: CLI commands
# ---------------------------------------------------------------------------

class TestCliHelp:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CARQ" in result.output

    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestStatusCommand:
    def test_status_table_output(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"id": "abc-123", "status": "done", "task_type": "PARSE_PDF"})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["status", "abc-123"])
        assert result.exit_code == 0

    def test_status_json_output(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"id": "abc-123", "status": "done"})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["status", "abc-123", "--output", "json"])
        assert result.exit_code == 0

    def test_status_connection_error(self):
        runner = CliRunner()
        with patch("carq.cli.client.make_request", side_effect=Exception("conn refused")):
            result = runner.invoke(cli, ["status", "abc-123"])
        assert result.exit_code != 0


class TestListCommand:
    def test_list_table_output(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"tasks": [
            {"id": "t1", "task_type": "PARSE_PDF", "status": "done",
             "priority": 0, "attempt_count": 1, "created_at": "2026-01-01T00:00:00"}
        ]})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0

    def test_list_json_output(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"tasks": []})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["list", "--output", "json"])
        assert result.exit_code == 0

    def test_list_with_status_filter(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"tasks": []})
        with patch("carq.cli.client.make_request", return_value=mock_resp) as mock_req:
            result = runner.invoke(cli, ["list", "--status-filter", "failed"])
        assert result.exit_code == 0


class TestRetryCommand:
    def test_retry_success(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"status": "queued"})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["retry", "abc-123"])
        assert result.exit_code == 0

    def test_retry_connection_error(self):
        runner = CliRunner()
        with patch("carq.cli.client.make_request", side_effect=Exception("server down")):
            result = runner.invoke(cli, ["retry", "abc-123"])
        assert result.exit_code != 0


class TestStatsCommand:
    def test_stats_table_output(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {
            "embeddings_generated": 1000,
            "total_tokens_used": 50000,
            "cache_hit_rate": 0.75,
        })
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0

    def test_stats_json_output(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"embeddings_generated": 100})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["stats", "--output", "json"])
        assert result.exit_code == 0


class TestHealthCommand:
    def test_health_ok(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"status": "healthy"})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["health"])
        assert result.exit_code == 0
        assert "healthy" in result.output

    def test_health_unhealthy(self):
        runner = CliRunner()
        mock_resp = _mock_response(200, {"status": "degraded"})
        with patch("carq.cli.client.make_request", return_value=mock_resp):
            result = runner.invoke(cli, ["health"])
        assert result.exit_code == 0

    def test_health_unreachable(self):
        runner = CliRunner()
        with patch("carq.cli.client.make_request", side_effect=Exception("timeout")):
            result = runner.invoke(cli, ["health"])
        assert result.exit_code == 1


class TestMakeRequest:
    def test_successful_request(self):
        mock_resp = _mock_response(200, {"ok": True})
        with patch("httpx.request", return_value=mock_resp):
            resp = make_request("GET", "/health")
        assert resp.status_code == 200

    def test_retries_on_server_error(self):
        """Should retry on 500, succeed on 3rd attempt."""
        import httpx
        fail_resp = _mock_response(500)
        ok_resp = _mock_response(200, {"ok": True})

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPStatusError("err", request=MagicMock(), response=fail_resp)
            return ok_resp

        with patch("httpx.request", side_effect=side_effect):
            with patch("time.sleep"):  # Don't actually sleep
                resp = make_request("GET", "/test", max_retries=3)
        assert resp.status_code == 200
        assert call_count == 3

    def test_raises_on_client_error_without_retry(self):
        """400 errors should not be retried."""
        import httpx
        fail_resp = _mock_response(400)

        with patch("httpx.request") as mock_req:
            mock_req.return_value = fail_resp
            fail_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "bad request", request=MagicMock(), response=fail_resp
            )
            with pytest.raises(httpx.HTTPStatusError):
                make_request("GET", "/test", max_retries=3)
