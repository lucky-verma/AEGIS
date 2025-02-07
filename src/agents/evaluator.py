import httpx
import logging
import json
from typing import Dict


class EvaluatorAgent:
    def __init__(self):
        self.base_url = "http://host.docker.internal:11434"
        self.model = "deepseek-r1:1.5b"
        self.logger = logging.getLogger(__name__)

    async def evaluate(self, query: str, answer: str) -> Dict:
        try:
            prompt = f"""Evaluate this answer for the query: {query}
            
            Answer: {answer}
            
            Provide a JSON response with:
            - accuracy_score (0-10)
            - completeness_score (0-10)
            - relevance_score (0-10)
            - final_verdict (approved/needs_review/rejected)"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt},
                )
                return self._parse_evaluation(response.json()["response"])
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            return {"error": str(e)}

    def _parse_evaluation(self, response: str) -> Dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "accuracy_score": 0,
                "completeness_score": 0,
                "relevance_score": 0,
                "final_verdict": "failed",
            }
