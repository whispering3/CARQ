"""Cliente CLI do CARQ para ingestão de documentos, monitoramento de status e gerenciamento do sistema."""
import json
import os
import sys
import time
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.json import JSON
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def _get_config() -> tuple[str, str, str]:
    """Lê as configurações de conexão do CARQ a partir do ambiente no momento da chamada (não na importação)."""
    return (
        os.getenv("CARQ_API_URL", "http://localhost:8000"),
        os.getenv("CARQ_API_KEY", ""),
        os.getenv("CARQ_API_TOKEN", ""),
    )


def get_headers():
    _, api_key, api_token = _get_config()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    elif api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    return headers

def make_request(method: str, path: str, max_retries: int = 3, **kwargs):
    """Realiza requisição HTTP com retry e backoff exponencial."""
    api_url, _, _ = _get_config()
    url = f"{api_url}{path}"
    for attempt in range(max_retries):
        try:
            response = httpx.request(method, url, headers=get_headers(), timeout=30.0, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            console.print(f"[yellow]Attempt {attempt+1} failed, retrying in {wait}s...[/yellow]")
            time.sleep(wait)
        except httpx.RequestError as e:
            if attempt == max_retries - 1:
                raise click.ClickException(f"Connection error: {e}")
            wait = 2 ** attempt
            time.sleep(wait)

@click.group()
@click.version_option(version="1.0.0", prog_name="carq")
def cli():
    """CARQ - Fila de Processamento RAG com Consciência de Contexto."""
    pass

@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--document-type",
    "-t",
    default="pdf",
    type=click.Choice(["pdf", "text", "url"]),
    help="Tipo de documento",
)
@click.option("--priority", "-p", default=0, type=int, help="Prioridade de processamento (maior = primeiro)")
@click.option("--metadata", "-m", default=None, help="String de metadados JSON")
@click.option("--wait", "-w", is_flag=True, help="Aguardar a conclusão do processamento")
@click.option("--output", "-o", type=click.Choice(["json", "table"]), default="table")
def ingest(file_path, document_type, priority, metadata, wait, output):
    """Ingere um documento para processamento."""
    path = Path(file_path)
    meta = json.loads(metadata) if metadata else {}

    payload = {
        "source_uri": path.absolute().as_uri(),
        "document_type": document_type,
        "priority": priority,
        "attributes": meta,
    }
    if document_type == "text":
        payload["content"] = path.read_text(encoding="utf-8")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Submitting document...", total=None)
        try:
            response = make_request("POST", "/api/v1/documents", json=payload)
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e))

    data = response.json()

    if wait:
        document_id = data.get("document_id")
        console.print(f"[green]Document submitted.[/green] Waiting for processing (document {document_id})...")
        _poll_until_done(document_id)
        return

    _print_result(data, output, title="Ingest Result")

@cli.command()
@click.argument("document_id")
@click.option("--output", "-o", type=click.Choice(["json", "table"]), default="table")
def status(document_id, output):
    """Obtém o status de um documento em processamento."""
    try:
        response = make_request("GET", f"/api/v1/documents/{document_id}")
    except Exception as e:
        raise click.ClickException(str(e))
    _print_result(response.json(), output, title=f"Document {document_id}")

@cli.command(name="list")
@click.option("--status-filter", "-s", default=None, help="Filtrar por status (pending, processing, done, failed)")
@click.option("--limit", "-l", default=20, type=int, help="Número máximo de resultados")
@click.option("--output", "-o", type=click.Choice(["json", "table"]), default="table")
def list_tasks(status_filter, limit, output):
    """Lista documentos em processamento."""
    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter
    try:
        response = make_request("GET", "/api/v1/documents", params=params)
    except Exception as e:
        raise click.ClickException(str(e))
    data = response.json()
    if output == "json":
        console.print(JSON(json.dumps(data)))
    else:
        _print_tasks_table(data.get("documents", []))

@cli.command()
@click.argument("document_id")
def retry(document_id):
    """Repete uma tarefa de pipeline de documento com falha."""
    try:
        make_request("POST", f"/api/v1/documents/{document_id}/retry")
    except Exception as e:
        raise click.ClickException(str(e))
    console.print(f"[green]Document {document_id} queued for retry.[/green]")

@cli.command()
@click.option("--output", "-o", type=click.Choice(["json", "table"]), default="table")
def stats(output):
    """Exibe estatísticas do sistema."""
    try:
        response = make_request("GET", "/api/v1/stats")
    except Exception as e:
        raise click.ClickException(str(e))
    _print_result(response.json(), output, title="System Statistics")

@cli.command()
def health():
    """Verifica a saúde do sistema."""
    try:
        response = make_request("GET", "/health/ready", max_retries=1)
        data = response.json()
        status_icon = "OK" if data.get("status") in {"healthy", "ready"} else "WARN"
        console.print(f"{status_icon} System status: [bold]{data.get('status', 'unknown')}[/bold]")
    except Exception as e:
        console.print(f"[red]❌ System unreachable: {e}[/red]")
        sys.exit(1)

def _poll_until_done(document_id: str, timeout: int = 300):
    """Consulta o status do documento até que seja concluído ou com falha."""
    start = time.time()
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Processing...", total=None)
        while time.time() - start < timeout:
            try:
                response = make_request("GET", f"/api/v1/documents/{document_id}", max_retries=1)
                data = response.json()
                current_status = data.get("status", "unknown")
                progress.update(task, description=f"Processing... (status: {current_status})")
                if current_status in ("done", "failed"):
                    icon = "✅" if current_status == "done" else "❌"
                    console.print(f"{icon} Document {document_id}: [bold]{current_status}[/bold]")
                    return
            except httpx.HTTPStatusError as e:
                console.print(
                    f"[yellow]Status check returned {e.response.status_code} "
                    f"for document {document_id}, retrying...[/yellow]"
                )
            except Exception:
                pass
            time.sleep(2)
    console.print(f"[yellow]Timeout waiting for document {document_id}[/yellow]")

def _print_result(data: dict, output: str, title: str = "Result"):
    if output == "json":
        console.print(JSON(json.dumps(data)))
    else:
        table = Table(title=title, show_header=True)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        for k, v in data.items():
            table.add_row(str(k), str(v))
        console.print(table)

def _print_tasks_table(tasks: list):
    table = Table(title="Tasks", show_header=True)
    for col in ["ID", "Type", "Status", "Created"]:
        table.add_column(col, style="cyan" if col == "ID" else "white")
    for t in tasks:
        table.add_row(
            str(t.get("id", ""))[:8] + "...",
            str(t.get("document_type", "")),
            str(t.get("status", "")),
            str(t.get("created_at", ""))[:19],
        )
    console.print(table)

if __name__ == "__main__":
    cli()
