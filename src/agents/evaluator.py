import httpx
import logging
from typing import Dict, List, Optional
import re
import time
import asyncio
from dataclasses import dataclass
from functools import lru_cache
from asyncio import Semaphore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetric:
    name: str
    score: float
    confidence: float
    details: str
    processing_time: float
    error: Optional[str] = None


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


class EvaluatorAgent:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "deepseek-r1:7b"
        self.logger = logging.getLogger(__name__)
        self.circuit_breaker = CircuitBreaker()
        self.semaphore = Semaphore(3)  # Limit concurrent model calls
        self.response_cache = {}
        self.metric_weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.20,
            "coherence": 0.15,
            "factual_consistency": 0.20,
        }
        logger.info(f"EvaluatorAgent initialized with model: {self.model}")

    async def evaluate(self, query: str, answer: str) -> Dict:
        total_start_time = time.time()
        logger.info(f"\n{'='*50}\nStarting evaluation for query: {query}\n{'='*50}")
        print("\nInitiating evaluation process")

        try:
            if not self._validate_input(query, answer):
                return self.get_default_evaluation()

            # Evaluate metrics concurrently with timeout
            try:
                metrics = await asyncio.wait_for(
                    self._evaluate_all_metrics(query, answer),
                    timeout=200,  # 2-minute timeout for all evaluations
                )
            except asyncio.TimeoutError:
                logger.error("Metric evaluation timed out")
                return self._generate_partial_evaluation(query, answer)

            # Generate verdict and feedback
            verdict = self._generate_verdict(metrics)
            feedback = await self._generate_feedback(metrics)

            total_time = time.time() - total_start_time
            result = self._compile_results(metrics, verdict, feedback, total_time)

            self._log_evaluation_summary(result)
            return result

        except Exception as e:
            logger.exception("Evaluation error occurred")
            print(f"\nERROR: Evaluation process failed: {str(e)}")
            return self.get_default_evaluation()

    async def _evaluate_all_metrics(
        self, query: str, answer: str
    ) -> Dict[str, EvaluationMetric]:
        logger.info("Starting concurrent metric evaluations")
        metrics = {}

        # Create tasks for all metric evaluations
        tasks = []
        for metric_name in self.metric_weights.keys():
            task = self._evaluate_single_metric(query, answer, metric_name)
            tasks.append(task)

        # Execute all evaluations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for metric_name, result in zip(self.metric_weights.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating {metric_name}: {str(result)}")
                metrics[metric_name] = self._create_error_metric(
                    metric_name, str(result)
                )
            else:
                metrics[metric_name] = result

        return metrics

    async def _evaluate_single_metric(
        self, query: str, answer: str, metric_name: str
    ) -> EvaluationMetric:
        start_time = time.time()
        print(f"\nEvaluating {metric_name}...")

        try:
            # Try to get cached response
            cache_key = f"{metric_name}:{hash(query)}:{hash(answer)}"
            cached_result = self._get_cached_response(cache_key)
            if cached_result:
                logger.info(f"Using cached result for {metric_name}")
                return cached_result

            prompt = self._build_metric_prompt(query, answer, metric_name)
            response = await self._get_model_response_with_timeout(prompt)

            if response.error:
                return self._create_error_metric(metric_name, response.error)

            metric = self._parse_metric_evaluation(
                metric_name, response.text, time.time() - start_time
            )

            # Cache successful result
            self._cache_response(cache_key, metric)

            logger.info(f"{metric_name.capitalize()} score: {metric.score:.2f}")
            print(f"{metric_name.capitalize()} evaluation complete: {metric.score:.2f}")

            return metric

        except Exception as e:
            logger.exception(f"Error evaluating {metric_name}")
            return self._create_error_metric(metric_name, str(e))

    @lru_cache(maxsize=100)
    def _get_cached_response(self, cache_key: str) -> Optional[EvaluationMetric]:
        return self.response_cache.get(cache_key)

    def _cache_response(self, cache_key: str, metric: EvaluationMetric):
        self.response_cache[cache_key] = metric

    async def _get_model_response_with_timeout(
        self, prompt: str, timeout: int = 30
    ) -> ModelResponse:
        if not self.circuit_breaker.can_proceed():
            return ModelResponse.error("Circuit breaker is open")

        try:
            async with self.semaphore:
                async with httpx.AsyncClient(timeout=timeout) as client:
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

                    self.circuit_breaker.record_success()
                    return ModelResponse(
                        response.json().get("response", ""), time.time()
                    )

        except asyncio.TimeoutError:
            self.circuit_breaker.record_failure()
            return ModelResponse.error(f"Timeout after {timeout} seconds")
        except Exception as e:
            self.circuit_breaker.record_failure()
            return ModelResponse.error(str(e))

    def _create_error_metric(
        self, metric_name: str, error_msg: str
    ) -> EvaluationMetric:
        return EvaluationMetric(
            name=metric_name,
            score=0.0,
            confidence=0.0,
            details=f"Error: {error_msg}",
            processing_time=0.0,
            error=error_msg,
        )

    def _build_metric_prompt(self, query: str, answer: str, metric: str) -> str:
        logger.debug(f"Building prompt for {metric} evaluation")

        metric_instructions = {
            "accuracy": """
                Evaluate the factual correctness of the answer.
                Consider:
                - Accuracy of stated facts
                - Precision of information
                - Absence of errors
                Provide a numerical score (0-1) and detailed reasoning.
            """,
            "completeness": """
                Assess how thoroughly the answer addresses the query.
                Consider:
                - Coverage of all aspects
                - Depth of explanation
                - Missing important information
                Provide a numerical score (0-1) and detailed reasoning.
            """,
            "relevance": """
                Evaluate how well the answer addresses the query intent.
                Consider:
                - Direct response to query
                - Focus on key points
                - Relevance of information
                Provide a numerical score (0-1) and detailed reasoning.
            """,
            "coherence": """
                Assess the logical flow and clarity of the answer.
                Consider:
                - Logical structure
                - Clear connections
                - Readability
                Provide a numerical score (0-1) and detailed reasoning.
            """,
            "factual_consistency": """
                Check for internal consistency and factual alignment.
                Consider:
                - Consistent statements
                - Logical contradictions
                - Factual accuracy
                Provide a numerical score (0-1) and detailed reasoning.
            """,
        }

        prompt = f"""Evaluate this answer based on {metric}:

        Query: {query}
        Answer: {answer}

        {metric_instructions[metric]}

        Format your response exactly as follows:
        <think>
        [Your detailed evaluation process]
        </think>

        Score: [single number between 0 and 1]
        Reasoning: [detailed explanation with specific examples]
        """

        return prompt

    def _parse_metric_evaluation(
        self, metric_name: str, response: str, processing_time: float
    ) -> EvaluationMetric:
        logger.debug(f"Parsing evaluation for {metric_name}")
        try:
            # Extract thinking process for confidence calculation
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            thinking_process = think_match.group(1) if think_match else ""

            # Extract score with improved regex
            score_match = re.search(r"Score:\s*(\d*\.?\d+)", response)
            if not score_match:
                logger.warning(
                    f"No score found for {metric_name}, using regex fallback"
                )
                # Fallback: try to find any number between 0 and 1
                score_match = re.search(r"\b(0|1|0\.\d+)\b", response)

            score = float(score_match.group(1)) if score_match else 0.0
            score = max(0.0, min(1.0, score))  # Ensure score is between 0 and 1

            # Extract reasoning
            reasoning_match = re.search(
                r"Reasoning:\s*(.+?)(?=<|$)", response, re.DOTALL
            )
            reasoning = (
                reasoning_match.group(1).strip()
                if reasoning_match
                else "No reasoning provided"
            )

            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(thinking_process, reasoning)

            logger.debug(
                f"{metric_name} evaluation parsed - Score: {score:.2f}, Confidence: {confidence:.2f}"
            )

            return EvaluationMetric(
                name=metric_name,
                score=score,
                confidence=confidence,
                details=reasoning,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error parsing {metric_name} evaluation: {str(e)}")
            return self._create_error_metric(metric_name, f"Parsing error: {str(e)}")

    def _calculate_confidence(self, thinking_process: str, reasoning: str) -> float:
        confidence = 0.5  # Base confidence

        # Analyze thinking process
        if thinking_process:
            word_count = len(thinking_process.split())
            if word_count > 50:
                confidence += 0.1
            if re.search(
                r"(compare|analyze|consider|evaluate)", thinking_process.lower()
            ):
                confidence += 0.1

        # Analyze reasoning
        if reasoning:
            if len(reasoning.split()) > 30:
                confidence += 0.1
            if re.search(
                r"(example|instance|specifically|for instance)", reasoning.lower()
            ):
                confidence += 0.1
            if re.search(r"(however|although|nevertheless|while)", reasoning.lower()):
                confidence += 0.1

        return min(confidence, 1.0)

    def _generate_verdict(self, metrics: Dict[str, EvaluationMetric]) -> str:
        logger.info("Generating final verdict")

        try:
            # Calculate weighted score
            weighted_score = sum(
                metrics[metric].score * weight
                for metric, weight in self.metric_weights.items()
                if not metrics[metric].error
            )

            # Calculate average confidence for valid metrics
            valid_metrics = [m for m in metrics.values() if not m.error]
            if not valid_metrics:
                return "failed"

            avg_confidence = sum(m.confidence for m in valid_metrics) / len(
                valid_metrics
            )

            # Determine verdict with more nuanced criteria
            if weighted_score >= 0.8 and avg_confidence >= 0.7:
                verdict = "approved"
            elif weighted_score >= 0.6 or (
                weighted_score >= 0.5 and avg_confidence >= 0.8
            ):
                verdict = "needs_review"
            else:
                verdict = "rejected"

            logger.info(
                f"Verdict: {verdict} (Score: {weighted_score:.2f}, Confidence: {avg_confidence:.2f})"
            )
            return verdict

        except Exception as e:
            logger.error(f"Error generating verdict: {str(e)}")
            return "failed"

    async def _generate_feedback(self, metrics: Dict[str, EvaluationMetric]) -> Dict:
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "improvement_priority": [],
        }

        try:
            # Analyze metrics and generate specific feedback
            for metric in metrics.values():
                if metric.error:
                    continue

                if metric.score >= 0.8:
                    feedback["strengths"].append(
                        {
                            "aspect": metric.name,
                            "score": metric.score,
                            "details": metric.details[:200],
                        }
                    )
                elif metric.score < 0.6:
                    feedback["weaknesses"].append(
                        {
                            "aspect": metric.name,
                            "score": metric.score,
                            "details": metric.details[:200],
                        }
                    )

                    # Generate specific improvement suggestions
                    suggestion = await self._generate_improvement_suggestion(metric)
                    if suggestion:
                        feedback["suggestions"].append(suggestion)

            # Prioritize improvements
            feedback["improvement_priority"] = self._prioritize_improvements(
                feedback["weaknesses"]
            )

            return feedback

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            return {
                "strengths": [],
                "weaknesses": ["Error generating detailed feedback"],
                "suggestions": ["Please try again"],
                "improvement_priority": [],
            }

    async def _generate_improvement_suggestion(
        self, metric: EvaluationMetric
    ) -> Optional[str]:
        try:
            prompt = f"""Based on this evaluation:
            Aspect: {metric.name}
            Score: {metric.score}
            Details: {metric.details}

            Provide a specific, actionable suggestion for improvement in one sentence."""

            response = await self._get_model_response_with_timeout(prompt, timeout=200)
            return response.text.strip() if not response.error else None

        except Exception:
            return None

    def _prioritize_improvements(self, weaknesses: List[Dict]) -> List[str]:
        return sorted(
            [w["aspect"] for w in weaknesses],
            key=lambda x: self.metric_weights.get(x, 0),
            reverse=True,
        )

    def _compile_results(
        self,
        metrics: Dict[str, EvaluationMetric],
        verdict: str,
        feedback: Dict,
        total_time: float,
    ) -> Dict:
        return {
            "scores": {f"{m.name}_score": int(m.score * 10) for m in metrics.values()},
            "confidence_scores": {
                f"{m.name}_confidence": m.confidence for m in metrics.values()
            },
            "final_verdict": verdict,
            "detailed_feedback": feedback,
            "processing_time": total_time,
            "error_metrics": [m.name for m in metrics.values() if m.error],
        }

    def _log_evaluation_summary(self, result: Dict):
        logger.info("\nEvaluation Summary:")
        logger.info(f"Verdict: {result['final_verdict']}")
        logger.info("Scores:")
        for metric, score in result["scores"].items():
            logger.info(f"- {metric}: {score}/10")
        if result["error_metrics"]:
            logger.warning(f"Failed metrics: {', '.join(result['error_metrics'])}")

    def _generate_partial_evaluation(self, query: str, answer: str) -> Dict:
        """Generate a partial evaluation when full evaluation fails"""
        logger.warning("Generating partial evaluation due to timeout")

        # Attempt quick evaluation of most important metrics
        try:
            quick_metrics = asyncio.run(
                asyncio.wait_for(
                    self._evaluate_critical_metrics(query, answer), timeout=200
                )
            )
            return self._compile_results(
                quick_metrics,
                "needs_review",
                {"weaknesses": ["Partial evaluation due to timeout"]},
                30.0,
            )
        except Exception:
            return self.get_default_evaluation()

    async def _evaluate_critical_metrics(
        self, query: str, answer: str
    ) -> Dict[str, EvaluationMetric]:
        """Evaluate only the most critical metrics with reduced complexity"""
        critical_metrics = ["accuracy", "relevance"]
        metrics = {}

        for metric_name in critical_metrics:
            metrics[metric_name] = await self._evaluate_single_metric(
                query, answer, metric_name
            )

        return metrics

    def _validate_input(self, query: str, answer: str) -> bool:
        """Validate input parameters for evaluation"""
        try:
            if not query or not answer:
                logger.warning("Empty query or answer")
                return False

            if len(query.strip()) < 3:
                logger.warning("Query too short")
                return False

            if len(answer.strip()) < 10:
                logger.warning("Answer too short")
                return False

            # Check for meaningful content
            meaningful_words = len(re.findall(r"\b\w{3,}\b", answer))
            if meaningful_words < 5:
                logger.warning("Answer lacks meaningful content")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            return False

    def get_default_evaluation(self) -> Dict:
        """Return default evaluation when processing fails"""
        logger.warning("Returning default evaluation")
        return {
            "scores": {
                "accuracy_score": 0,
                "completeness_score": 0,
                "relevance_score": 0,
                "coherence_score": 0,
                "factual_consistency_score": 0,
            },
            "confidence_scores": {
                "accuracy_confidence": 0.0,
                "completeness_confidence": 0.0,
                "relevance_confidence": 0.0,
                "coherence_confidence": 0.0,
                "factual_consistency_confidence": 0.0,
            },
            "final_verdict": "failed",
            "detailed_feedback": {
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "suggestions": ["Try again with a different query"],
                "improvement_priority": [],
            },
            "processing_time": 0.0,
            "error_metrics": ["all"],
            "error_message": "Evaluation process failed",
        }

    def _generate_partial_evaluation(self, query: str, answer: str) -> Dict:
        """Generate partial evaluation when full evaluation fails"""
        logger.warning("Generating partial evaluation")
        try:
            # Quick evaluation of basic metrics
            basic_scores = {
                "accuracy_score": self._quick_accuracy_check(answer),
                "completeness_score": self._quick_completeness_check(query, answer),
                "relevance_score": self._quick_relevance_check(query, answer),
            }

            return {
                "scores": {
                    **basic_scores,
                    "coherence_score": 0,
                    "factual_consistency_score": 0,
                },
                "confidence_scores": {metric: 0.5 for metric in basic_scores.keys()},
                "final_verdict": "needs_review",
                "detailed_feedback": {
                    "strengths": [],
                    "weaknesses": ["Partial evaluation only"],
                    "suggestions": ["Complete evaluation recommended"],
                    "improvement_priority": [],
                },
                "processing_time": 0.0,
                "error_metrics": ["coherence", "factual_consistency"],
                "partial_evaluation": True,
            }
        except Exception as e:
            logger.error(f"Error in partial evaluation: {str(e)}")
            return self.get_default_evaluation()

    def _quick_accuracy_check(self, answer: str) -> int:
        """Quick basic check for answer accuracy"""
        try:
            # Basic checks for answer quality
            if len(answer.split()) < 10:
                return 2
            if len(answer.split()) > 50:
                return 5
            return 3
        except Exception:
            return 0

    def _quick_completeness_check(self, query: str, answer: str) -> int:
        """Quick basic check for answer completeness"""
        try:
            # Check query terms coverage
            query_terms = set(re.findall(r"\b\w+\b", query.lower()))
            answer_terms = set(re.findall(r"\b\w+\b", answer.lower()))
            coverage = len(query_terms.intersection(answer_terms)) / len(query_terms)
            return int(coverage * 10)
        except Exception:
            return 0

    def _quick_relevance_check(self, query: str, answer: str) -> int:
        """Quick basic check for answer relevance"""
        try:
            # Basic relevance check based on term overlap
            query_terms = set(re.findall(r"\b\w+\b", query.lower()))
            answer_terms = set(re.findall(r"\b\w+\b", answer.lower()))
            overlap = len(query_terms.intersection(answer_terms))
            return min(int((overlap / len(query_terms)) * 10), 10)
        except Exception:
            return 0
