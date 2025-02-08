import httpx
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ReasonerAgent:
    def __init__(self):
        self.base_url = "http://ollama:11434"
        self.model = "deepseek-r1"

    async def reason(self, query: str, contexts: List[Dict]) -> str:
        try:
            prompt = self._build_reasoning_prompt(query, contexts)

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "options": {"temperature": 0.3, "num_ctx": 4096, "top_p": 0.9},
                    },
                )
                response.raise_for_status()
                raw_response = response.json()
                logger.info(f"Raw Ollama response:\n{raw_response}")
                return self._parse_reasoning(raw_response["response"])
        except Exception as e:
            logger.exception(f"Reasoning error: {str(e)}")
            return "Could not generate answer due to an error."

    def _parse_reasoning(self, response: str) -> str:
        logger.info(f"Parsing reasoning from:\n{response}")
        parts = response.split("</think>")
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            return response.strip()

    def _build_reasoning_prompt(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"""Perform multi-step reasoning for query: {query}
        
        Contexts:
        {self._format_contexts(contexts)}
        
        Follow these steps:
        1. Analyze each context for relevant information
        2. Identify connections between different contexts
        3. Synthesize a comprehensive answer
        4. Highlight any remaining uncertainties
        
        Final Answer:"""
        return prompt

    def _format_contexts(self, contexts: List[Dict]) -> str:
        return "\n".join(
            f"Context {i+1} ({ctx['url']}):\n{ctx['content'][:200]}..."
            for i, ctx in enumerate(contexts)
        )
