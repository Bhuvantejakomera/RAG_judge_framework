from __future__ import annotations

import src.runtime_bootstrap  # noqa: F401

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.status import Status


REQUIRED_COLUMNS = {"s_wiki_title", "question", "possible_answers"}

console = Console()


def _abort(message: str) -> None:
    console.print(f"[bold red]ERROR:[/bold red] {message}")
    sys.exit(1)


def _check_python_version() -> None:
    if sys.version_info < (3, 12):
        _abort("Python 3.12 or higher is required.")
    console.print(f"[green]Python version:[/green] {platform.python_version()}")


def _check_dataset() -> None:
    dataset_path = Path.cwd() / "data" / "test.tsv"
    if not dataset_path.exists():
        _abort("Missing data/test.tsv. Please add the benchmark dataset.")

    with dataset_path.open("r", encoding="utf-8") as handle:
        headers = handle.readline().strip().split("\t")
    missing = REQUIRED_COLUMNS - set(headers)
    if missing:
        _abort(f"Dataset is missing required columns: {', '.join(missing)}")
    console.print("[green]Dataset check passed.[/green]")


def _check_env() -> None:
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        console.print("[yellow]WARNING:[/yellow] GEMINI_API_KEY missing. Gemini will fall back to Ollama if needed.")
    else:
        console.print("[green]Gemini API key found.[/green]")


def _check_ollama() -> None:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, check=True, text=True, timeout=20)
    except FileNotFoundError:
        _abort("Ollama executable not found. Install Ollama and ensure it is on the PATH.")
    except subprocess.CalledProcessError:
        console.print("[yellow]Unable to query Ollama; retrying with service check.[/yellow]")
        _check_ollama_service()
        return

    if "qwen2.5:3b-instruct-q4_K_M" not in result.stdout:
        console.print("[yellow]Target model not present. Pulling qwen2.5:3b-instruct-q4_K_M...[/yellow]")
        subprocess.run(["ollama", "pull", "qwen2.5:3b-instruct-q4_K_M"], check=True)
    console.print("[green]Ollama is ready.[/green]")


def _check_ollama_service() -> None:
    try:
        response = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"], capture_output=True, check=True, text=True, timeout=10
        )
        tags = json.loads(response.stdout)
        if not tags:
            _abort("Ollama service responded but no models are available.")
    except Exception:
        _abort("Ollama service is not running. Start Ollama locally or ensure the HTTP API is accessible.")


def _activate_venv() -> None:
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        console.print("[yellow]WARNING:[/yellow] You are not running inside a virtual environment.")
    else:
        console.print("[green]Virtual environment active.[/green]")


def _main() -> None:
    with Status("Verifying environment", spinner="dots") as status:
        status.update("Checking Python version...")
        _check_python_version()
        status.update("Verifying virtual environment...")
        _activate_venv()
        status.update("Checking dataset...")
        _check_dataset()
        status.update("Loading environment variables...")
        _check_env()
        status.update("Checking Ollama runtime...")
        _check_ollama()

    console.print("[bold green]Pre-flight checks complete.[/bold green]")
    console.print("[bold blue]Launching PopQA-RAG-Judge...[/bold blue]")
    from main import main as launch_app

    launch_app()


if __name__ == "__main__":
    _main()
