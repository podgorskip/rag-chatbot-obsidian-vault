import re
import logging
import pandas as pd
from pathlib import Path
import argparse
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def clean_markdown(text: str) -> str:
    text = re.sub(r"^---[\s\S]*?---\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\*{1,3}([^\*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_by_heading(title: str, content: str, min_length: int = 100) -> list[dict]:
    sections = re.split(r"\n(?=#{1,3} )", content)
    chunks = []

    for section in sections:
        lines = section.strip().splitlines()
        heading = lines[0].lstrip("#").strip() if lines else ""
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section.strip()
        body = clean_markdown(body)

        if len(body) < min_length:
            continue

        chunk_title = f"{title} > {heading}" if heading else title
        chunks.append({"title": chunk_title, "content": body})

    return chunks


def read_vault(
        vault_path: str,
        exclude_folders: list[str] | None = None,
        exclude_files: list[str] | None   = None,
    ) -> list[dict]:

    exclude_folders = set(exclude_folders or ["templates", ".trash", ".obsidian"])
    exclude_files   = set(exclude_files   or [])

    vault   = Path(vault_path)
    records = []

    for md_file in vault.rglob("*.md"):
        if any(part in exclude_folders for part in md_file.parts):
            continue

        if md_file.name in exclude_files:
            continue

        try:
            raw   = md_file.read_text(encoding="utf-8")
            title = md_file.stem
            rel   = str(md_file.relative_to(vault))

            chunks = chunk_by_heading(title, raw)

            if not chunks:
                cleaned = clean_markdown(raw)
                if cleaned:
                    chunks = [{"title": title, "content": cleaned}]

            for chunk in chunks:
                chunk["source"] = rel
                records.append(chunk)

        except Exception as e:
            log.warning(f"Could not read {md_file}: {e}")

    log.info(f"Loaded {len(records)} chunks from {vault_path}")
    return records

def build_knowledge_base(
        vault_path: str,
        output_path: str = "knowledge_base.pkl",
        model_name: str = "all-MiniLM-L6-v2",
        exclude_folders: list[str] | None = None,
        exclude_files:   list[str] | None = None
    ) -> pd.DataFrame:

    records = read_vault(vault_path, exclude_folders, exclude_files)

    if not records:
        raise ValueError(f"No markdown files found in: {vault_path}")

    log.info(f"Loading embedding model: {model_name}")
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    model = SentenceTransformer(model_name)

    df = pd.DataFrame(records)
    contents = df["content"].tolist()

    log.info(f"Embedding {len(contents)} chunks...")
    embeddings = model.encode(
        contents,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64
    )

    df["embedding"] = list(embeddings)

    df.to_pickle(output_path)
    log.info(f"Saved knowledge base → {output_path}  ({len(df)} chunks)")

    return df