import httpx
import logging
import json
from typing import Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class EvaluatorAgent:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "deepseek-r1:1.5b"
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def evaluate(self, query: str, answer: str) -> Dict:
        try:
            prompt = self.build_evaluation_prompt(query, answer)
            print("\n=== EVALUATION PROMPT ===")
            print(prompt)
            print("=====================\n")

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_ctx": 4096},
                    },
                )
                response.raise_for_status()

                raw_response = response.json()
                response_text = raw_response.get("response", "")
                return self.parse_evaluation_response(response_text)

        except Exception as e:
            self.logger.exception("Evaluation error occurred")
            print(f"\n=== EVALUATION ERROR ===\n{str(e)}\n====================\n")
            return self.get_default_evaluation()

    def build_evaluation_prompt(self, query: str, answer: str) -> str:
        return f"""Evaluate the following answer for the given query:

                Query: {query}
                Answer: {answer}

                Instructions:
                Analyze the answer and provide scores based on these criteria:
                1. Accuracy: How factually correct is the answer? (0-10)
                2. Completeness: How thoroughly does it address the query? (0-10)
                3. Relevance: How well does it match the query intent? (0-10)

                Your response should follow this format:
                <think>
                [Your evaluation reasoning here]
                </think>

                Accuracy Score: [score]
                Completeness Score: [score]
                Relevance Score: [score]
                Final Verdict: [approved/needs_review/rejected]
                """

    def parse_evaluation_response(self, response: str) -> Dict:
        print("\n=== PARSING EVALUATION ===")
        print(f"Raw response:\n{response}")

        try:
            # Extract content after </think> tag
            parts = response.split("</think>")
            evaluation_text = parts[-1].strip() if len(parts) > 1 else response.strip()

            print(f"\nExtracted evaluation text:\n{evaluation_text}")

            # Initialize result dictionary
            result = {
                "accuracy_score": 0,
                "completeness_score": 0,
                "relevance_score": 0,
                "final_verdict": "needs_review",
            }

            # Parse each line
            for line in evaluation_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = [part.strip() for part in line.split(":", 1)]
                    key = key.lower().replace(" ", "_")

                    if key in [
                        "accuracy_score",
                        "completeness_score",
                        "relevance_score",
                    ]:
                        try:
                            # Extract numeric value
                            numeric_value = "".join(filter(str.isdigit, value))
                            result[key] = int(numeric_value) if numeric_value else 0
                        except ValueError:
                            result[key] = 0
                    elif key == "final_verdict":
                        verdict = value.lower().strip()
                        if verdict in ["approved", "needs_review", "rejected"]:
                            result["final_verdict"] = verdict
                        else:
                            result["final_verdict"] = "needs_review"

            print(f"\nParsed result:\n{json.dumps(result, indent=2)}")
            print("=====================\n")

            return result

        except Exception as e:
            self.logger.exception(f"Error parsing evaluation response {str(e)}")
            return self.get_default_evaluation()

    def get_default_evaluation(self) -> Dict:
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "relevance_score": 0,
            "final_verdict": "failed",
        }
