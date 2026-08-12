from __future__ import annotations

from src.embeddings.embedder import EmbeddingEngine
from src.generator.ollama_client import LocalOllamaGenerator


def test_truncates_long_texts_before_embedding() -> None:
    engine = EmbeddingEngine(model_name="BAAI/bge-small-en-v1.5")
    long_text = "word " * 2000

    prepared = engine._prepare_texts([long_text])[0]
    token_count = len(engine._tokenizer.encode(prepared, add_special_tokens=False))

    assert token_count <= engine.max_sequence_length
    assert prepared != long_text


def test_build_prompt_truncates_large_context() -> None:
    generator = LocalOllamaGenerator()
    large_context = "alpha beta gamma " * 5000

    prompt = generator._build_prompt("What is this?", large_context)

    assert len(prompt) < len(large_context) + 2000
    assert "[truncated]" in prompt
