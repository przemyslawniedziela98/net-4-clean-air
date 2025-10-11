import os
from typing import List, Dict, Any
import time 
import openai

from app.services.embedder import Embedder
from app.services.qdrant_wrapper import QdrantWrapper
from app.services.prometheus import metrics
from app.logger import logger


openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatService:
    def __init__(self, qdrant: QdrantWrapper, embedder: Embedder, model: str = "gpt-3.5-turbo"):
        """
        Initialize the ChatService.

        Args:
            qdrant (QdrantWrapper): Wrapper for the Qdrant vector database.
            embedder (Embedder): Text embedding service.
            model (str): OpenAI chat model to use. Defaults to 'gpt-3.5-turbo'.
        """
        self.qdrant = qdrant
        self.embedder = embedder
        self.model = model

    def _embed_query(self, question: str) -> Any:
        """Compute the embedding for a question and record metrics."""
        try:
            logger.info("Embedding query for question.")
            return self.embedder.embed([question])[0]
        except Exception as e:
            metrics.CHAT_ERRORS.labels(model=self.model, stage="embedding").inc()
            logger.exception("Error during embedding")
            raise

    def _search_qdrant(self, query_emb: Any, top_k: int) -> List[Dict[str, Any]]:
        """Perform a Qdrant search with metrics and logging."""
        try:
            qdrant_start = time.perf_counter()
            results = self.qdrant.search(query_emb, top_k=top_k)
            duration = time.perf_counter() - qdrant_start
            metrics.QDRANT_SEARCH_LATENCY.observe(duration)
            logger.info(f"Qdrant search completed in {duration:.3f}s with {len(results)} results")
            return results
        except Exception as e:
            metrics.CHAT_ERRORS.labels(model=self.model, stage="qdrant").inc()
            logger.exception("Error during Qdrant search")
            raise

    def _build_prompt(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Build the full prompt text for the OpenAI API call."""
        context_text = self._build_context(results)
        return (
            f"Answer the following question based on the provided literature.\n\n"
            f"Literature:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def _generate_answer(self, prompt: str, max_tokens: int) -> str:
        """Send the prompt to OpenAI, record metrics, and return the answer."""
        try:
            openai_start = time.perf_counter()
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            duration = time.perf_counter() - openai_start
            metrics.OPENAI_LATENCY.labels(model=self.model).observe(duration)
            logger.info(f"OpenAI completion finished in {duration:.3f}s")
            return response.choices[0].message.content.strip()
        except Exception as e:
            metrics.CHAT_ERRORS.labels(model=self.model, stage="openai").inc()
            logger.exception("Error during OpenAI completion")
            return f"Error: {e}"

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Helper: Build a textual context from Qdrant search results."""
        context_parts = []
        for r in results:
            payload = r.get("payload", {})
            title = payload.get("TITLE OF THE PAPER", "")
            aim = payload.get("AIM OF THE PAPER", "")
            findings = payload.get("MAIN FINDINGS OF THE PAPER", "")
            context_parts.append(f"Title: {title}\nAim: {aim}\nFindings: {findings}")
        return "\n\n".join(context_parts)

    def answer_question(self, question: str, top_k: int = 5, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Answer a user question using retrieved literature and an OpenAI chat model.

        Args:
            question (str): The user's question.
            top_k (int): Number of top relevant documents to use for context.
            max_tokens (int): Max tokens for the OpenAI completion.

        Returns:
            Dict[str, Any]: Dict with 'answer' and 'context_docs'.
        """
        metrics.CHAT_REQUESTS.labels(model=self.model).inc()
        total_start = time.perf_counter()
        logger.info(f"Answering question with top_k={top_k}: {question}")

        query_emb = self._embed_query(question)
        results = self._search_qdrant(query_emb, top_k)
        prompt = self._build_prompt(question, results)
        answer = self._generate_answer(prompt, max_tokens)

        total_duration = time.perf_counter() - total_start
        metrics.CHAT_LATENCY.labels(model=self.model).observe(total_duration)
        logger.info(f"Total ChatService duration: {total_duration:.3f}s")

        return {"answer": answer, "context_docs": results}