import os
from typing import List, Dict, Any
import openai

from app.services.embedder import Embedder
from app.services.qdrant_wrapper import QdrantWrapper
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

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Build a textual context from Qdrant search results.

        Args:
            results (List[Dict[str, Any]]): Search results containing document payloads.

        Returns:
            str: Concatenated context text for the prompt.
        """
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
        logger.info(f"Answering question with top_k={top_k}: {question}")

        query_emb = self.embedder.embed([question])[0]
        results = self.qdrant.search(query_emb, top_k=top_k)
        logger.debug(f"Retrieved {len(results)} context documents")

        context_text = self._build_context(results)
        prompt = (
            f"Answer the following question based on the provided literature.\n\n"
            f"Literature:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated ChatGPT answer")
        except Exception as e:
            logger.exception(f"Error during ChatGPT call: {e}")
            answer = f"Error: {e}"

        return {"answer": answer, "context_docs": results}
