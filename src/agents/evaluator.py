import httpx
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self):
        self.base_url = "http://ollama:11434"
        self.model = "deepseek-r1"

    async def evaluate(self, query: str, answer: str) -> Dict:
        try:
            prompt = f"""Evaluate this answer for the query: {query}
            
            Answer: {answer}
            
            Provide an evaluation with:
            - accuracy_score (0-10)
            - completeness_score (0-10)
            - relevance_score (0-10)
            - final_verdict (approved/needs_review/rejected)

            Format your response as:
            Accuracy Score: [score]
            Completeness Score: [score]
            Relevance Score: [score]
            Final Verdict: [verdict]
            """

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt},
                )
                raw_response = response.json()
                logger.info(f"Raw Ollama evaluation response:\n{raw_response}")
                return self._parse_evaluation(raw_response["response"])
        except Exception as e:
            logger.exception(f"Evaluation error: {str(e)}")
            return {"error": str(e)}

    def _parse_evaluation(self, response: str) -> Dict:
        logger.info(f"Parsing evaluation from:\n{response}")
        # Extract content after the last </think> tag
        parts = response.split("</think>")
        if len(parts) > 1:
            evaluation_text = parts[-1].strip()
        else:
            evaluation_text = response.strip()

        # Parse the evaluation text
        result = {}
        for line in evaluation_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if key in ["accuracy_score", "completeness_score", "relevance_score"]:
                    result[key] = int(value)
                elif key == "final_verdict":
                    result[key] = value

        if not result:
            self.logger.error(f"Failed to parse evaluation: {evaluation_text}")
            result = {
                "accuracy_score": 0,
                "completeness_score": 0,
                "relevance_score": 0,
                "final_verdict": "failed",
            }

        return result
