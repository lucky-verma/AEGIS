import httpx
import logging
import json
from typing import List, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class ReasonerAgent:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "deepseek-r1:1.5b"
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def reason(self, query: str, contexts: List[Dict]) -> str:
        try:
            prompt = self.build_reasoning_prompt(query, contexts)
            print("\n=== REASONING PROMPT ===")
            print(prompt)
            print("=====================\n")

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_ctx": 4096, "top_p": 0.9},
                    },
                )
                response.raise_for_status()
                raw_response = response.json()
                response_text = raw_response.get("response", "")
                return self.parse_reasoning_response(response_text)

        except Exception as e:
            self.logger.exception("Reasoning error occurred")
            print(f"\n=== REASONING ERROR ===\n{str(e)}\n===================\n")
            return "Could not generate answer due to an error."

    def build_reasoning_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_text = self.format_contexts(contexts)

        prompt = f"""Query: {query}

            Available Information:
            {context_text}

            Instructions:
            1. Analyze the provided information carefully
            2. Identify key relevant details
            3. Synthesize a comprehensive answer
            4. Ensure the response is factual and well-structured

            Your response should follow this format:
            <think>
            [Your reasoning process here]
            </think>

            [Your final answer here]
            """
        return prompt

    def format_contexts(self, contexts: List[Dict]) -> str:
        formatted_contexts = []
        for idx, ctx in enumerate(contexts, 1):
            content = ctx.get("content", "").strip()
            url = ctx.get("url", "No URL")
            title = ctx.get("title", "No Title")

            formatted_ctx = f"""Source {idx}:
                    Title: {title}
                    URL: {url}
                    Content: {content[:100]}..."""
            formatted_contexts.append(formatted_ctx)

        return "\n\n".join(formatted_contexts)

    def parse_reasoning_response(self, response: str) -> str:
        print("\n=== PARSING REASONING RESPONSE ===")
        print(f"Raw response:\n{response}")

        try:
            # Split on </think> and take the last part
            parts = response.split("</think>")
            if len(parts) > 1:
                final_answer = parts[-1].strip()
            else:
                # If no </think> tag, try to extract a meaningful response
                final_answer = response.strip()

            print(f"\nParsed answer:\n{final_answer}")
            print("============================\n")

            return final_answer

        except Exception as e:
            self.logger.exception(f"Error parsing reasoning response: {str(e)}")
            return "Error parsing the response"
