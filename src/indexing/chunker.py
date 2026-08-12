from __future__ import annotations

from typing import List

from src.models.data_models import DocumentChunk
from src.runtime_bootstrap import configure_native_threading


class TokenChunker:
    """Chunks text into token-aware spans using a sliding window."""

    def __init__(self, tokenizer_name: str = "BAAI/bge-small-en-v1.5", chunk_size: int = 400, overlap: int = 80) -> None:
        configure_native_threading()
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, article_title: str, text: str, source_url: str = "") -> List[DocumentChunk]:
        """Split text into overlapping chunks based on token counts."""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(token_ids)
        chunks: List[DocumentChunk] = []
        start = 0
        chunk_number = 1

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = token_ids[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{article_title}-{chunk_number}",
                    article_title=article_title,
                    chunk_number=chunk_number,
                    text=chunk_text,
                    source_url=source_url,
                    token_count=len(chunk_tokens),
                )
            )
            chunk_number += 1
            if end == total_tokens:
                break
            start += self.chunk_size - self.overlap

        return chunks
