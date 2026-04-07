from typing import Any
from torch import Tensor
from rags.llm_client import LLMClient
from rags.rag_config import Config
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rags.utils import estimate_tokens

class RAG:
  def __init__(
        self,
        client: LLMClient,
        embedding_model: SentenceTransformer,
        df: pd.DataFrame,
        config: Config = Config(),
        llm_model: str='llama3'
      ):

    self.client = client
    self.embedding_model = embedding_model
    self.df = df.copy()
    self.config = config
    self.llm_model = llm_model

    self.cumulative_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
        }

  def _prepare_chunks(self, candidates: pd.DataFrame) -> list[dict[str, int | Any]]:
      chunks = []
      if candidates.empty:
          return chunks
      top_sim = candidates.iloc[0]['similarity']
      current_tokens = 0

      for row in candidates.itertuples(index=False):
          curr_sim = row.similarity
          delta = abs(top_sim - curr_sim)

          if delta > self.config.delta_cutoff:
              break

          estimated_tokens = estimate_tokens(row.content)
          if current_tokens + estimated_tokens > self.config.max_context_tokens:
              break

          chunks.append({
              "title": row.title,
              "content": row.content,
              "similarity": curr_sim,
              "length": len(row.content)
          })

          current_tokens += estimated_tokens

      return chunks

  def _track_tokens(self, usage) -> None:
      if self.client.provider != "openai":
          return
      if usage:
          self.cumulative_tokens["prompt_tokens"] += usage.prompt_tokens
          self.cumulative_tokens["completion_tokens"] += usage.completion_tokens
          self.cumulative_tokens["total_tokens"] += usage.total_tokens

  def embed_query(self, query: str) -> Tensor:
    return self.embedding_model.encode(query, normalize_embeddings=True)

  def _compute_embedding_matrix(self) -> np.ndarray:
      if not hasattr(self, '_embedding_matrix'):
          self._embedding_matrix = np.stack(self.df['embedding'])
      return self._embedding_matrix

  def _semantic_scores(self, query_embedding: Tensor) -> np.ndarray:
      return np.dot(self._compute_embedding_matrix(), query_embedding)

  def _bm25_scores(self, query: str) -> np.ndarray:
      if not hasattr(self, '_bm25'):
          tokenized_corpus = [str(text).lower().split() for text in self.df['content']]
          self._bm25 = BM25Okapi(tokenized_corpus)

      tokenized_query = query.lower().split()
      scores = self._bm25.get_scores(tokenized_query)
      max_score = scores.max()
      return scores / max_score if max_score > 0 else scores

  def retrieve(self, query: str, query_embedding: Tensor) -> list:
      semantic = self._semantic_scores(query_embedding)
      bm25 = self._bm25_scores(query)
      combined = (
              self.config.semantic_weight * semantic +
              self.config.bm25_weight * bm25
      )

      self.df['similarity'] = combined
      threshold = self.df['similarity'].quantile(1 - self.config.top_fraction)
      candidates = self.df[
          (self.df['similarity'] >= threshold) &
          (self.df['similarity'] > self.config.min_similarity)
      ]
      candidates = candidates.sort_values(by='similarity', ascending=False)

      if candidates.empty:
          return []

      return self._prepare_chunks(candidates)

  def process_query(self, question: str, history: str, prompt_template: str) -> str:
      prompt = prompt_template.format(history=history, question=question)
      response = self.client.chat.completions.create(
          model=self.llm_model,
          messages=[{"role": "user", "content": prompt}]
      )
      self._track_tokens(response.usage)
      return response.choices[0].message.content.strip()

  def build_context(self, chunks: list) -> str:
    context_text = ""
    current_context_tokens = 0

    for chunk in chunks:
      chunk_text = f"PASSAGE ({chunk['title']}): '{chunk['content']}'\n"
      chunk_tokens = estimate_tokens(chunk_text)

      if current_context_tokens + chunk_tokens <= self.config.max_context_tokens:
        context_text += chunk_text
        current_context_tokens += chunk_tokens
      else:
        logging.info(f"Context tokens exceeded (~{current_context_tokens}).")
        break

    return context_text