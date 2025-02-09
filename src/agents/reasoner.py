import httpx
import logging
import json
from typing import List, Dict, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dataclasses import dataclass
import asyncio
import re
import time
from functools import lru_cache
from asyncio import Semaphore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    step_type: str
    content: str
    confidence: float
    metadata: Dict
    processing_time: float


class ModelResponse:
    def __init__(self, text: str, processing_time: float):
        self.text = text
        self.processing_time = processing_time
        self.error = None

    @classmethod
    def error(cls, error_msg: str):
        response = cls("", 0.0)
        response.error = error_msg
        return response


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        if time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
            return True
        return False


class ReasonerAgent:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "deepseek-r1:1.5b"
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.reasoning_steps = [
            self._extract_key_information,
            self._identify_patterns,
            self._synthesize_answer,
            self._self_reflect,
        ]
        self.circuit_breaker = CircuitBreaker()
        self.semaphore = Semaphore(3)  # Limit concurrent model calls
        self.response_cache = {}
        logger.info(f"ReasonerAgent initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    )
    async def reason(self, query: str, contexts: List[Dict]) -> str:
        total_start_time = time.time()
        logger.info(
            f"\n{'='*50}\nStarting reasoning process for query: {query}\n{'='*50}"
        )
        print(f"\nInitiating reasoning process for: {query}")

        try:
            if not self._validate_input(query, contexts):
                return "Invalid input provided"

            reasoning_chain = []
            current_context = self._preprocess_contexts(contexts)

            # Execute reasoning steps with timeout
            for step_idx, step_func in enumerate(self.reasoning_steps, 1):
                try:
                    step_result = await asyncio.wait_for(
                        step_func(query, current_context, reasoning_chain),
                        timeout=200,  #  2 minutes timeout per step
                    )

                    if not step_result:
                        logger.error(f"Step {step_idx} failed")
                        continue

                    self._log_step_metrics(step_result)
                    reasoning_chain.append(step_result)
                    current_context = self._update_context(current_context, step_result)

                except asyncio.TimeoutError:
                    logger.error(f"Step {step_idx} timed out")
                    print(f"Step {step_idx} timed out, continuing with partial results")
                    continue

            # Generate final answer with timeout
            try:
                final_answer = await asyncio.wait_for(
                    self._generate_final_answer(query, reasoning_chain), timeout=200
                )
            except asyncio.TimeoutError:
                logger.error("Final answer generation timed out")
                final_answer = self._generate_fallback_answer(reasoning_chain)

            total_time = time.time() - total_start_time
            self._log_completion_metrics(total_time, reasoning_chain)
            return final_answer

        except Exception as e:
            logger.exception("Reasoning error occurred")
            print(f"\nERROR: Reasoning process failed: {str(e)}")
            return "Could not generate answer due to an error."

    def _validate_input(self, query: str, contexts: List[Dict]) -> bool:
        if not query or not contexts:
            logger.error("Empty query or contexts")
            return False
        if len(query.strip()) < 3:
            logger.error("Query too short")
            return False
        if not any(ctx.get("content") for ctx in contexts):
            logger.error("No valid content in contexts")
            return False
        return True

    @lru_cache(maxsize=100)
    def _get_cached_response(self, prompt: str) -> Optional[ModelResponse]:
        return self.response_cache.get(prompt)

    async def _get_model_response(self, prompt: str) -> ModelResponse:
        if not self.circuit_breaker.can_proceed():
            return ModelResponse.error("Circuit breaker is open")

        # Check cache first
        cached_response = self._get_cached_response(prompt)
        if cached_response:
            logger.info("Using cached response")
            return cached_response

        start_time = time.time()
        try:
            async with self.semaphore:
                async with httpx.AsyncClient(timeout=200.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "num_ctx": 4096,
                                "top_p": 0.9,
                            },
                        },
                    )
                    response.raise_for_status()

                    processing_time = time.time() - start_time
                    response_text = response.json().get("response", "")

                    model_response = ModelResponse(response_text, processing_time)
                    self.response_cache[prompt] = model_response
                    self.circuit_breaker.record_success()

                    return model_response

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Model request failed: {str(e)}")
            return ModelResponse.error(str(e))

    async def _extract_key_information(
        self, query: str, contexts: Dict, chain: List[ReasoningStep]
    ) -> ReasoningStep:
        start_time = time.time()
        logger.info("Starting key information extraction")

        try:
            prompt = f"""Extract key information for query: {query}

            Context:
            {self._format_contexts(contexts)}

            Instructions:
            1. Identify and list key facts, dates, names, and numbers
            2. Note any uncertainties or contradictions
            3. Focus on most relevant information to query

            Format your response as:
            <think>
            [Your detailed extraction process]
            </think>

            Key Information:
            - [Key point 1]
            - [Key point 2]
            etc.
            """

            response = await self._get_model_response_with_timeout(prompt, timeout=200)
            if response.error:
                return self._create_error_step("extraction", response.error)

            extracted_info = self._parse_extraction(response.text)
            confidence = self._calculate_confidence(response.text)
            processing_time = time.time() - start_time

            return ReasoningStep(
                step_type="extraction",
                content=extracted_info,
                confidence=confidence,
                metadata={"source_count": len(contexts)},
                processing_time=processing_time,
            )

        except Exception as e:
            logger.exception("Error in information extraction")
            return self._create_error_step("extraction", str(e))

    async def _get_model_response_with_timeout(
        self, prompt: str, timeout: int = 30
    ) -> ModelResponse:
        try:
            return await asyncio.wait_for(
                self._get_model_response(prompt), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Model response timed out after {timeout} seconds")
            return ModelResponse.error(f"Timeout after {timeout} seconds")
        except Exception as e:
            logger.exception("Error getting model response")
            return ModelResponse.error(str(e))

    def _create_error_step(self, step_type: str, error_msg: str) -> ReasoningStep:
        return ReasoningStep(
            step_type=step_type,
            content=f"Error: {error_msg}",
            confidence=0.0,
            metadata={"error": error_msg},
            processing_time=0.0,
        )

    async def _identify_patterns(
        self, query: str, contexts: Dict, chain: List[ReasoningStep]
    ) -> ReasoningStep:
        start_time = time.time()
        logger.info("Starting pattern identification")

        try:
            previous_step = chain[-1] if chain else None
            if previous_step and previous_step.confidence < 0.3:
                logger.warning(
                    "Low confidence in previous step, using simplified pattern analysis"
                )
                return await self._simplified_pattern_analysis(query, contexts)

            prompt = f"""Analyze patterns in the following information:

            Query: {query}
            Previous Analysis: {previous_step.content if previous_step else 'None'}

            Instructions:
            1. Identify recurring themes and concepts
            2. Note relationships between facts
            3. Highlight any temporal or causal patterns

            Format your response as:
            <think>
            [Your pattern analysis process]
            </think>

            Patterns Found:
            1. [Pattern description]
            2. [Pattern description]
            etc.
            """

            response = await self._get_model_response_with_timeout(prompt, timeout=200)
            if response.error:
                return self._create_error_step("pattern_analysis", response.error)

            patterns = self._parse_patterns(response.text)
            confidence = self._calculate_confidence(response.text)
            processing_time = time.time() - start_time

            return ReasoningStep(
                step_type="pattern_analysis",
                content=patterns,
                confidence=confidence,
                metadata={"patterns_found": len(patterns.split("\n"))},
                processing_time=processing_time,
            )

        except Exception as e:
            logger.exception("Error in pattern identification")
            return self._create_error_step("pattern_analysis", str(e))

    async def _simplified_pattern_analysis(
        self, query: str, contexts: Dict
    ) -> ReasoningStep:
        """Fallback method for pattern analysis when previous step has low confidence"""
        start_time = time.time()
        prompt = f"""Provide a simple analysis of the main themes related to: {query}
        Context: {self._format_contexts(contexts)[:1000]}
        
        List the main themes found:"""

        response = await self._get_model_response_with_timeout(prompt, timeout=200)
        processing_time = time.time() - start_time

        return ReasoningStep(
            step_type="pattern_analysis",
            content=response.text,
            confidence=0.5,
            metadata={"simplified": True},
            processing_time=processing_time,
        )

    async def _synthesize_answer(
        self, query: str, contexts: Dict, chain: List[ReasoningStep]
    ) -> ReasoningStep:
        start_time = time.time()
        logger.info("Starting answer synthesis")

        try:
            extracted_info = chain[0].content if len(chain) > 0 else ""
            patterns = chain[1].content if len(chain) > 1 else ""

            prompt = f"""Generate a clear, direct answer to this query: {query}

            Available Information:
            {extracted_info}

            Identified Patterns:
            {patterns}

            Instructions:
            1. Provide a direct, factual answer
            2. Focus only on information supported by the sources
            3. Be concise but complete
            4. If information is missing or uncertain, acknowledge it
            5. Do not include phrases like "based on the information" or "according to"

            Format your response as a clear, direct paragraph. Do not include any meta-text or thinking process.
            """

            response = await self._get_model_response_with_timeout(prompt, timeout=45)
            if response.error:
                return self._create_error_step("synthesis", response.error)

            # Clean the response
            answer = self._clean_answer(response.text)
            confidence = self._calculate_confidence(response.text)
            completeness = self._assess_completeness(answer, query)

            return ReasoningStep(
                step_type="synthesis",
                content=answer,
                confidence=confidence,
                metadata={
                    "completeness": completeness,
                    "word_count": len(answer.split()),
                },
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.exception("Error in answer synthesis")
            return self._create_error_step("synthesis", str(e))

    def _clean_answer(self, response: str) -> str:
        """Clean the response to get only the direct answer"""
        # Remove thinking process
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        # Remove common prefixes
        prefixes = [
            "Here's what I found:",
            "Based on the information",
            "According to the sources",
            "The answer is:",
            "To answer your question,",
        ]
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix) :].strip()

        # Clean up whitespace and formatting
        response = re.sub(r"\s+", " ", response).strip()

        return response

    def _parse_synthesis(self, response: str) -> str:
        """Parse the synthesis response to extract the final answer"""
        try:
            # First try to extract content after "Synthesized Answer:"
            parts = response.split("Synthesized Answer:")
            if len(parts) > 1:
                return parts[-1].strip()

            # If that fails, try to extract content after </think>
            parts = response.split("</think>")
            if len(parts) > 1:
                return parts[-1].strip()

            # If both fail, return the cleaned response
            return response.strip()
        except Exception as e:
            logger.error(f"Error parsing synthesis: {str(e)}")
            return "Error parsing synthesized answer"

    def _assess_completeness(self, synthesis: str, query: str) -> float:
        """Assess how completely the synthesis addresses the query"""
        try:
            completeness = 0.0

            # Check query term coverage
            query_terms = set(re.findall(r"\w+", query.lower()))
            synthesis_terms = set(re.findall(r"\w+", synthesis.lower()))
            term_coverage = len(query_terms.intersection(synthesis_terms)) / len(
                query_terms
            )
            completeness += term_coverage * 0.4

            # Check length
            if len(synthesis.split()) >= 50:
                completeness += 0.3

            # Check structure indicators
            if re.search(r"\b(however|moreover|furthermore|therefore)\b", synthesis):
                completeness += 0.2

            # Check for specific details
            if re.search(r"\b(specifically|for example|such as)\b", synthesis):
                completeness += 0.1

            return min(completeness, 1.0)
        except Exception as e:
            logger.error(f"Error assessing completeness: {str(e)}")
            return 0.0

    def _generate_fallback_answer(self, reasoning_chain: List[ReasoningStep]) -> str:
        """Generate a simple answer when final answer generation fails"""
        try:
            # Get the most confident step
            best_step = max(reasoning_chain, key=lambda x: x.confidence)

            if best_step.confidence > 0.5:
                return f"Based on available information: {best_step.content[:500]}..."
            else:
                return "Unable to generate a complete answer due to processing issues."
        except Exception:
            return "Error generating answer."

    def _validate_reasoning_chain(self, chain: List[ReasoningStep]) -> bool:
        """Validate the reasoning chain for completeness and quality"""
        if not chain:
            return False

        required_steps = {"extraction", "pattern_analysis", "synthesis"}
        completed_steps = {step.step_type for step in chain}

        if not required_steps.issubset(completed_steps):
            return False

        avg_confidence = sum(step.confidence for step in chain) / len(chain)
        return avg_confidence >= 0.5

    def _log_completion_metrics(self, total_time: float, chain: List[ReasoningStep]):
        """Log detailed metrics about the reasoning process"""
        metrics = {
            "total_time": total_time,
            "steps_completed": len(chain),
            "average_confidence": sum(step.confidence for step in chain) / len(chain),
            "step_times": {step.step_type: step.processing_time for step in chain},
        }
        logger.info(f"Reasoning metrics: {json.dumps(metrics, indent=2)}")

        return metrics

    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for a response"""
        confidence = 0.5  # Base confidence

        try:
            # Check for thinking process
            if "<think>" in response and "</think>" in response:
                confidence += 0.2

            # Check response length
            if len(response.split()) > 50:
                confidence += 0.1

            # Check for specific details
            if re.search(r"\b(specifically|for example|such as)\b", response.lower()):
                confidence += 0.1

            # Check for analytical language
            if re.search(r"\b(because|therefore|thus|hence)\b", response.lower()):
                confidence += 0.1

            return min(confidence, 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _parse_extraction(self, response: str) -> str:
        """Parse the extraction response"""
        try:
            parts = response.split("Key Information:")
            return parts[-1].strip() if len(parts) > 1 else response.strip()
        except Exception as e:
            logger.error(f"Error parsing extraction: {str(e)}")
            return "Error parsing extraction"

    def _parse_patterns(self, response: str) -> str:
        """Parse the patterns response"""
        try:
            parts = response.split("Patterns Found:")
            return parts[-1].strip() if len(parts) > 1 else response.strip()
        except Exception as e:
            logger.error(f"Error parsing patterns: {str(e)}")
            return "Error parsing patterns"

    async def _self_reflect(
        self, query: str, contexts: Dict, chain: List[ReasoningStep]
    ) -> ReasoningStep:
        start_time = time.time()
        logger.info("Starting self-reflection")

        try:
            synthesis = next(
                (step.content for step in chain if step.step_type == "synthesis"), ""
            )
            if not synthesis:
                return self._create_error_step(
                    "reflection", "No synthesis found for reflection"
                )

            prompt = f"""Reflect on this answer:

            Query: {query}
            Generated Answer: {synthesis}

            Instructions:
            1. Evaluate logical consistency
            2. Check factual accuracy
            3. Assess completeness
            4. Identify potential biases or gaps

            Format your response as:
            <think>
            [Your reflection process]
            </think>

            Reflection:
            1. Consistency: [assessment]
            2. Accuracy: [assessment]
            3. Completeness: [assessment]
            4. Gaps/Biases: [assessment]
            """

            response = await self._get_model_response_with_timeout(prompt, timeout=200)
            if response.error:
                return self._create_error_step("reflection", response.error)

            reflection = self._parse_reflection(response.text)
            confidence = self._calculate_confidence(response.text)
            needs_revision = self._needs_revision(reflection)
            processing_time = time.time() - start_time

            return ReasoningStep(
                step_type="reflection",
                content=reflection,
                confidence=confidence,
                metadata={
                    "needs_revision": needs_revision,
                    "synthesis_length": len(synthesis),
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.exception("Error in self-reflection")
            return self._create_error_step("reflection", str(e))

    def _preprocess_contexts(self, contexts: List[Dict]) -> Dict:
        """Preprocess and combine contexts for reasoning"""
        logger.info("Preprocessing contexts")
        try:
            combined = {"content": "", "metadata": {}, "sources": []}

            for ctx in contexts:
                # Add content with source marker
                source_id = len(combined["sources"]) + 1
                combined[
                    "content"
                ] += f"\nSource {source_id}:\n{ctx.get('content', '')}\n"

                # Collect metadata
                combined["metadata"].update(ctx.get("metadata", {}))

                # Store source information
                combined["sources"].append(
                    {
                        "id": source_id,
                        "title": ctx.get("title", "Untitled"),
                        "url": ctx.get("url", ""),
                        "relevance": ctx.get("relevance", 0.0),
                    }
                )

            logger.info(f"Preprocessed {len(contexts)} contexts")
            return combined

        except Exception as e:
            logger.error(f"Error preprocessing contexts: {str(e)}")
            return {"content": "", "metadata": {}, "sources": []}

    def _log_step_metrics(self, step_result: ReasoningStep):
        """Log detailed metrics for each reasoning step"""
        logger.info(
            f"""
        Step Metrics:
        - Type: {step_result.step_type}
        - Confidence: {step_result.confidence:.2f}
        - Processing Time: {step_result.processing_time:.2f}s
        - Content Length: {len(step_result.content)}
        - Metadata: {json.dumps(step_result.metadata, indent=2)}
        """
        )

    def _update_context(
        self, current_context: Dict, step_result: ReasoningStep
    ) -> Dict:
        """Update context with results from the current reasoning step"""
        try:
            updated_context = current_context.copy()

            # Add step results to context
            step_key = f"step_{step_result.step_type}"
            updated_context[step_key] = {
                "content": step_result.content,
                "confidence": step_result.confidence,
                "metadata": step_result.metadata,
            }

            # Update metadata
            updated_context["metadata"].update(
                {
                    f"{step_key}_completed": True,
                    f"{step_key}_confidence": step_result.confidence,
                }
            )

            return updated_context

        except Exception as e:
            logger.error(f"Error updating context: {str(e)}")
            return current_context

    async def _generate_final_answer(
        self, query: str, reasoning_chain: List[ReasoningStep]
    ) -> str:
        """Generate the final answer based on the reasoning chain"""
        start_time = time.time()
        logger.info("Generating final answer")

        try:
            # Get synthesis and reflection
            synthesis = next(
                (
                    step.content
                    for step in reasoning_chain
                    if step.step_type == "synthesis"
                ),
                "",
            )
            reflection = next(
                (
                    step.content
                    for step in reasoning_chain
                    if step.step_type == "reflection"
                ),
                "",
            )

            if not synthesis:
                return "Could not generate answer: No synthesis available"

            # Check if revision is needed
            if reflection and self._needs_revision(reflection):
                logger.info("Generating revised answer based on reflection")

                revision_prompt = f"""Revise this answer based on reflection:

                Original Query: {query}
                Original Answer: {synthesis}
                Reflection: {reflection}

                Instructions:
                1. Address the identified issues
                2. Maintain accurate information
                3. Improve clarity and completeness

                Format your response as:
                <think>
                [Your revision process]
                </think>

                Revised Answer:
                [Your improved answer]
                """

                response = await self._get_model_response_with_timeout(
                    revision_prompt, timeout=200
                )
                if not response.error:
                    final_answer = self._parse_final_answer(response.text)
                else:
                    final_answer = synthesis  # Fallback to original synthesis
            else:
                final_answer = synthesis

            logger.info(
                f"Final answer generated in {time.time() - start_time:.2f} seconds"
            )
            return final_answer

        except Exception as e:
            logger.exception(f"Error generating final answer {str(e)}")
            return "Error generating final answer"

    def _format_contexts(self, contexts: Dict) -> str:
        """Format contexts for prompt construction"""
        try:
            formatted_text = ""

            # Add main content
            if "content" in contexts:
                content = contexts["content"]
                # Truncate if too long
                if len(content) > 2000:
                    content = content[:1997] + "..."
                formatted_text += content

            # Add source information if available
            if "sources" in contexts:
                formatted_text += "\n\nSources:\n"
                for source in contexts["sources"]:
                    formatted_text += f"- {source.get('title', 'Untitled')} ({source.get('url', 'No URL')})\n"

            return formatted_text.strip()

        except Exception as e:
            logger.error(f"Error formatting contexts: {str(e)}")
            return "Error formatting contexts"

    def _parse_reflection(self, response: str) -> str:
        """Parse the reflection response"""
        try:
            parts = response.split("Reflection:")
            return parts[-1].strip() if len(parts) > 1 else response.strip()
        except Exception as e:
            logger.error(f"Error parsing reflection: {str(e)}")
            return "Error parsing reflection"

    def _needs_revision(self, reflection: str) -> bool:
        """Determine if the answer needs revision based on reflection"""
        try:
            # Look for negative indicators
            negative_indicators = [
                "incorrect",
                "incomplete",
                "inconsistent",
                "biased",
                "unclear",
                "missing",
                "error",
                "wrong",
                "gap",
            ]

            reflection_lower = reflection.lower()
            needs_revision = any(
                indicator in reflection_lower for indicator in negative_indicators
            )

            if needs_revision:
                found_indicators = [
                    ind for ind in negative_indicators if ind in reflection_lower
                ]
                logger.info(f"Revision needed. Found indicators: {found_indicators}")
            else:
                logger.info("No revision needed")

            return needs_revision

        except Exception as e:
            logger.error(f"Error checking revision need: {str(e)}")
            return False

    def _parse_final_answer(self, response: str) -> str:
        """Parse the final answer from the response"""
        try:
            # Try to extract content after "Revised Answer:" or "Final Answer:"
            for marker in ["Revised Answer:", "Final Answer:"]:
                parts = response.split(marker)
                if len(parts) > 1:
                    return parts[-1].strip()

            # Remove thinking process if present
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

            return response.strip()

        except Exception as e:
            logger.error(f"Error parsing final answer: {str(e)}")
            return "Error parsing final answer"
