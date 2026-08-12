from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import unquote

import httpx
import pandas as pd
import wikipediaapi
from wikipediaapi.exceptions import (
    WikiConnectionError,
    WikiHttpError,
    WikiHttpTimeoutError,
    WikiInvalidJsonError,
    WikiRateLimitError,
)

from src.config.settings import ConfigManager


class WikipediaFetcher:
    """Fetches and caches Wikipedia article content for dataset titles."""

    def __init__(self, data_directory: str | Path | None = None, config: Optional[ConfigManager] = None) -> None:
        self._config_manager = config or ConfigManager()
        self._settings = self._config_manager.settings
        self._cache_dir = Path(data_directory or Path.cwd() / "data" / "wikipedia_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

        user_agent = getattr(self._settings.wikipedia, "user_agent", "PopQARAGJudge/1.0 (research@example.com)")
        language = getattr(self._settings.wikipedia, "language", "en")

        self._wikipedia = wikipediaapi.Wikipedia(
            user_agent,
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            timeout=5.0,
            max_retries=1,
            retry_wait=1.0,
        )
        self._missing_log_path = Path(Path.cwd() / "logs" / "missing_articles.log")
        self._missing_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, title: str) -> str:
        sanitized = title.replace("/", "_").replace("\\", "_").strip()
        return "".join(ch for ch in sanitized if ch.isalnum() or ch in " ._- ") or "article"

    def _cache_path(self, title: str) -> Path:
        return self._cache_dir / f"{self._safe_filename(title)}.txt"

    def _log_missing(self, title: str, detail: str) -> None:
        with self._missing_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{title}\t{detail}\n")

    def _normalize_title(self, title: str) -> str:
        normalized = unquote(str(title))
        normalized = normalized.replace("_", " ").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _title_variants(self, title: str) -> list[str]:
        normalized = title.strip()
        variants: list[str] = []
        if not normalized:
            return variants

        variants.append(normalized)
        if " " in normalized:
            variants.append(normalized.replace(" ", "_"))
        if "_" in normalized:
            variants.append(normalized.replace("_", " "))

        if "(" not in normalized:
            suffixes = [
                " (film)",
                " (TV film)",
                " (TV series)",
                " (soundtrack)",
                " (novel)",
                " (song)",
            ]
            for suffix in suffixes:
                variants.append(f"{normalized}{suffix}")
                if " " in normalized:
                    variants.append(f"{normalized.replace(' ', '_')}{suffix}")

        seen: set[str] = set()
        deduped: list[str] = []
        for candidate in variants:
            candidate = candidate.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def _load_from_cache(self, title: str) -> str | None:
        cache_file = self._cache_path(title)
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def _save_to_cache(self, title: str, content: str) -> None:
        self._cache_path(title).write_text(content, encoding="utf-8")

    def _fetch_article(self, title: str) -> str | None:
        normalized_title = self._normalize_title(title)
        variants = self._title_variants(normalized_title)
        last_reason = "missing"

        for candidate in variants:
            try:
                page = self._wikipedia.page(candidate)
                if page.exists():
                    text = page.text or ""
                    if text.strip():
                        if candidate != normalized_title:
                            self._logger.info(
                                "Fetched fallback title '%s' for requested '%s'",
                                candidate,
                                title,
                            )
                        return text
                    last_reason = f"empty content for {candidate}"
                    continue
                self._logger.debug("Wikipedia article candidate missing: %s", candidate)
                last_reason = f"missing candidate {candidate}"
            except (
                WikiHttpTimeoutError,
                WikiConnectionError,
                WikiRateLimitError,
                WikiHttpError,
                WikiInvalidJsonError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as exc:
                self._logger.warning("Wikipedia request failed for %s: %s", candidate, exc)
                last_reason = str(exc)
            except Exception as exc:
                self._logger.exception(
                    "Unexpected error fetching Wikipedia article %s: %s",
                    candidate,
                    exc,
                )
                last_reason = str(exc)

        self._logger.warning(
            "Wikipedia article permanently missing for title '%s' after trying variants: %s",
            title,
            variants,
        )
        self._log_missing(normalized_title, last_reason)
        return None

    def _extract_titles(self, tsv_path: Path) -> Iterator[str]:
        yield from self.extract_unique_titles(tsv_path)

    def extract_unique_titles(self, tsv_path: str | Path) -> list[str]:
        dataset_path = Path(tsv_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Input file not found: {dataset_path}")

        try:
            df = pd.read_csv(dataset_path, sep="\t", dtype=str, usecols=["s_wiki_title"], keep_default_na=False)
        except ValueError as exc:
            raise ValueError("Dataset TSV must contain an 's_wiki_title' column.") from exc

        raw_titles = df["s_wiki_title"].dropna().astype(str).tolist()
        normalized_titles: list[str] = []
        seen: set[str] = set()
        for title in raw_titles:
            normalized = self._normalize_title(title)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_titles.append(normalized)
        return normalized_titles

    def get_unique_titles(self, dataset_tsv: str | Path) -> list[str]:
        return self.extract_unique_titles(dataset_tsv)

    def fetch_article(self, title: str) -> str | None:
        normalized_title = self._normalize_title(title)
        content = self._load_from_cache(normalized_title)
        if content is not None:
            return content

        content = self._fetch_article(normalized_title)
        if content is not None:
            self._save_to_cache(normalized_title, content)
        return content

    # Preserve older public interface for callers that still use fetch_title.
    fetch_title = fetch_article

    def fetch_all_titles(self, dataset_tsv: str | Path) -> dict[str, str]:
        """Fetch and cache Wikipedia content for unique titles in the dataset.

        Args:
            dataset_tsv: Path to the TSV file containing s_wiki_title values.

        Returns:
            Mapping of title to article text for successfully retrieved articles.
        """
        dataset_path = Path(dataset_tsv)
        result: dict[str, str] = {}
        titles = self.extract_unique_titles(dataset_path)

        for title in titles:
            try:
                content = self._load_from_cache(title)
                if content is None:
                    content = self._fetch_article(title)
                    if content is None:
                        continue
                    self._save_to_cache(title, content)
                result[title] = content
            except Exception as exc:
                self._logger.exception("Failed to fetch or cache article %s: %s", title, exc)
                self._log_missing(title, str(exc))
        return result
