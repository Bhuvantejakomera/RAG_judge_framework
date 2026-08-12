from __future__ import annotations

from pathlib import Path

from src.storage.vector_store import VectorStore


TARGET_TITLES = [
    "Sixteen Candles (film)",
    "Doomwatch (film)",
    "Into the Woods",
]


def main() -> None:
    index_dir = Path("index")
    store = VectorStore(index_dir)
    if not store.exists():
        raise FileNotFoundError("Index directory is missing or incomplete. Rebuild the index first.")

    store.load()
    article_map = store.get_article_chunk_map()
    found = {title: title in article_map for title in TARGET_TITLES}

    print("Index coverage verification")
    print("===========================")
    print(f"Index size: {store.index_size()}")
    print(f"Total metadata entries: {store.get_metadata_count()}")
    print("")
    for title in TARGET_TITLES:
        if found[title]:
            count = len(article_map[title])
            print(f"✔ {title}: present with {count} chunk(s)")
        else:
            print(f"✘ {title}: missing from article mapping")

    missing_titles = [title for title, present in found.items() if not present]
    if missing_titles:
        print("\nWARNING: Some key PopQA subject titles are not present in the index mapping.")
        print("Please rebuild the index with --rebuild-index or run the index builder directly.")
    else:
        print("\nAll target titles are present in the index mapping.")


if __name__ == "__main__":
    main()
