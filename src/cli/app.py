from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.config.settings import ConfigManager
from src.history.session_manager import SessionManager
from src.indexing.index_builder import IndexBuilder
from src.models.data_models import ConversationTurn
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import DenseFAISSRetriever
from src.storage.vector_store import VectorStore
from src.generator.ollama_client import LocalOllamaGenerator


class CLIApp:
    """Interactive CLI application for PopQA-RAG-Judge."""

    def __init__(self) -> None:
        self._console = Console()
        self._logger = logging.getLogger(__name__)
        self._settings = ConfigManager().settings
        self._session_manager = SessionManager()
        self._vector_store = VectorStore()
        self._index_builder = IndexBuilder(vector_store=self._vector_store)
        self._retriever = DenseFAISSRetriever(vector_store=self._vector_store)
        self._rewriter = QueryRewriter()
        self._generator = LocalOllamaGenerator()
        self._show_context = True
        self._started = False
        self._latencies: list[float] = []
        self._generation_latencies: list[float] = []

    def run(self, force_rebuild: bool = False) -> None:
        self._print_header()
        self._prepare_index(force_rebuild=force_rebuild)
        self._started = True
        self._interactive_loop()

    def run_evaluation(self, force_rebuild: bool = False, sample_size: int | None = None) -> None:
        """Run a one-off evaluation without entering the interactive loop."""
        self._print_header()
        self._prepare_index(force_rebuild=force_rebuild)
        self._command_evaluate(sample_size=sample_size)

    def _print_header(self) -> None:
        header = Panel(
            "[bold green]PopQA-RAG-Judge[/bold green]\nInteractive question answering with local retrieval and judge evaluation.",
            title="PopQA-RAG-Judge",
        )
        self._console.print(header)

    def _prepare_index(self, force_rebuild: bool = False) -> None:
        self._console.print("[blue]Starting Wikipedia knowledge base construction...[/blue]")
        if self._index_builder._vector_store.exists() and not force_rebuild:
            self._console.print("[green]Existing FAISS index detected. Loading index...[/green]")
        elif force_rebuild:
            self._console.print("[yellow]Force rebuilding FAISS index from dataset...[/yellow]")
        else:
            self._console.print("[yellow]No index found. Building index automatically from dataset...[/yellow]")

        self._index_builder.build_or_load(force_rebuild=force_rebuild)
        self._console.print("[green]Indexing complete! Saved FAISS index.[/green]\n")

    def _interactive_loop(self) -> None:
        while True:
            user_input = Prompt.ask("[bold cyan]User[/bold cyan]")
            normalized = user_input.strip().lower()
            if normalized in {"exit", "quit"}:
                self._console.print("[bold magenta]Goodbye![/bold magenta]")
                sys.exit(0)
            if normalized.startswith("--"):
                self._handle_command(normalized)
                continue
            self._handle_question(user_input)

    def _handle_command(self, command: str) -> None:
        parts = command.strip().split()
        cmd = parts[0].lower() if parts else ""

        if cmd == "--evaluate":
            sample_size = None
            if len(parts) >= 2:
                for token in parts[1:]:
                    cleaned = token.lstrip("-")
                    if cleaned.isdigit():
                        sample_size = int(cleaned)
                        break
            self._command_evaluate(sample_size=sample_size)
            return

        handlers = {
            "--help": self._command_help,
            "--stats": self._command_stats,
            "--hide-context": self._command_hide_context,
            "--show-context": self._command_show_context,
            "--clear-history": self._command_clear_history,
            "--rebuild-index": self._command_rebuild_index,
        }
        handler = handlers.get(cmd)
        if handler:
            handler()
        else:
            self._console.print(f"[red]Unknown command: {command}[/red]")

    def _handle_question(self, question: str) -> None:
        history = self._session_manager.get_sliding_history()
        rewritten = self._rewriter.rewrite(question, history)
        start_retrieval = time.perf_counter()
        retrieved = self._retriever.retrieve(rewritten, self._settings.retrieval.top_k)
        retrieval_latency = time.perf_counter() - start_retrieval
        self._latencies.append(retrieval_latency)
        context = self._build_context(retrieved)
        if self._show_context:
            self._display_retrieved_chunks(retrieved)
        start_generation = time.perf_counter()
        answer = self._generator.generate(rewritten, context)
        generation_latency = time.perf_counter() - start_generation
        self._generation_latencies.append(generation_latency)
        self._display_answer(answer, retrieval_latency, generation_latency)
        turn = ConversationTurn(
            question=question,
            rewritten_question=rewritten,
            retrieved_chunks=retrieved,
            generated_answer=answer,
            retrieval_time=retrieval_latency,
            generation_time=generation_latency,
        )
        self._session_manager.add_turn(turn)

    def _build_context(self, retrieved: Iterable[any]) -> str:
        return "\n\n".join(item.chunk.text for item in retrieved)

    def _display_retrieved_chunks(self, retrieved: Iterable[any]) -> None:
        table = Table(title="Top Retrieved Chunks", show_lines=True)
        table.add_column("Rank", style="bold")
        table.add_column("Title")
        table.add_column("Score")
        table.add_column("Preview")
        for idx, result in enumerate(retrieved[: self._settings.retrieval.display_k], start=1):
            preview = result.chunk.text[:200].replace("\n", " ")
            table.add_row(str(idx), result.chunk.article_title, f"{result.score:.4f}", preview)
        self._console.print(table)

    def _display_answer(self, answer: str, retrieval_latency: float, generation_latency: float) -> None:
        panel = Panel(
            f"[bold yellow]Answer[/bold yellow]\n{answer}\n\n"
            f"[green]Retrieval latency:[/green] {retrieval_latency:.3f}s\n"
            f"[green]Generation latency:[/green] {generation_latency:.3f}s",
            title="Result",
        )
        self._console.print(panel)

    def _command_help(self) -> None:
        table = Table(title="Available commands", show_lines=True)
        table.add_column("Command", style="bold cyan")
        table.add_column("Description")
        table.add_row("--help", "Show this help text.")
        table.add_row("--stats", "Display index and session statistics.")
        table.add_row("--hide-context", "Hide retrieved chunk previews.")
        table.add_row("--show-context", "Show retrieved chunk previews.")
        table.add_row("--clear-history", "Clear the current conversation memory.")
        table.add_row("--evaluate", "Run the evaluation pipeline.")
        table.add_row("--rebuild-index", "Rebuild the FAISS index from dataset.")
        table.add_row("exit / quit", "Exit the application.")
        self._console.print(table)

    def _command_stats(self) -> None:
        index_size = self._index_builder._vector_store.index_size()
        avg_retrieval = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        avg_generation = sum(self._generation_latencies) / len(self._generation_latencies) if self._generation_latencies else 0.0
        table = Table(title="System Statistics")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Session ID", str(self._session_manager.session_root.name))
        table.add_row("Total chunks", str(index_size))
        table.add_row("Embedding model", self._settings.embedding.model)
        table.add_row("Generator model", self._settings.llm.local_model)
        table.add_row("Average retrieval latency", f"{avg_retrieval:.3f}s")
        table.add_row("Average generation latency", f"{avg_generation:.3f}s")
        self._console.print(table)

    def _command_hide_context(self) -> None:
        self._show_context = False
        self._console.print("[green]Context display disabled.[/green]")

    def _command_show_context(self) -> None:
        self._show_context = True
        self._console.print("[green]Context display enabled.[/green]")

    def _command_clear_history(self) -> None:
        self._session_manager.clear_history()
        self._console.print("[green]Conversation history cleared.[/green]")

    def _command_evaluate(self, sample_size: int | None = None) -> None:
        self._run_evaluation_pipeline(sample_size=sample_size)

    def _run_evaluation_pipeline(self, sample_size: int | None = None) -> None:
        self._console.print("[blue]Running evaluation pipeline...[/blue]")
        from src.evaluation.pipeline import EvaluationPipeline

        EvaluationPipeline(
            session_manager=self._session_manager,
            retriever=self._retriever,
            rewriter=self._rewriter,
            generator=self._generator,
            sample_size=sample_size,
        ).run(evaluate=True, sample_size=sample_size)
        self._console.print("[green]Evaluation complete.[/green]")

    def _command_rebuild_index(self) -> None:
        confirmation = Prompt.ask("This will wipe and rebuild the index from dataset. Continue? [Y/N]", default="N")
        if confirmation.strip().lower() == "y":
            self._console.print("[yellow]Clearing existing index files...[/yellow]")
            self._index_builder._vector_store.clear()
            self._console.print("[yellow]Rebuilding index...[/yellow]")
            self._index_builder.build_or_load(force_rebuild=True)
            self._console.print("[green]Index rebuilt successfully.[/green]")
        else:
            self._console.print("[cyan]Rebuild cancelled.[/cyan]")
