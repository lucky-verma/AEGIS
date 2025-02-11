from collections import Counter
import httpx
import asyncio
from typing import List, Dict, Optional, Set

import spacy
from spacy.tokens import Token

from utils.search import SearxNGWrapper
import logging
from dataclasses import asdict, dataclass
from urllib.parse import urlparse
import re
import time
from functools import lru_cache
from asyncio import Semaphore
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from haystack.utils import ComponentDevice
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

device = ComponentDevice.from_str("cuda:0")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize spaCy and set up custom extensions
nlp = spacy.load("en_core_web_sm")
Token.set_extension("synsets", default=[])


@dataclass
class ProcessedDocument:
    title: str
    url: str
    content: str
    doc_type: str
    metadata: Dict
    relevance_score: float
    processing_time: float


class CircuitBreaker:

    def __init__(self, failure_threshold=10, reset_timeout=120):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False

    def record_failure(self):
        current_time = time.time()
        if current_time - self.last_failure_time > self.reset_timeout:
            self.reset()

        self.failure_count += 1
        self.last_failure_time = current_time

        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened! Failures: {self.failure_count}")

    def record_success(self):
        self.reset()
        logger.info("Circuit breaker reset")

    def reset(self):
        self.failure_count = 0
        self.is_open = False

    def status(self):
        return {
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "time_since_last_failure": time.time() - self.last_failure_time,
        }

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True

        if time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
            return True

        return False


@dataclass
class SubQuery:
    text: str
    type: str
    priority: int


@dataclass
class RefinedQuery:
    text: str
    confidence: float
    source_contexts: List[str]


class HybridQueryExpander:
    def __init__(self):
        self.nlp = nlp
        self.generator = OllamaGenerator(
            model="llama3.2:3b", url="http://localhost:11434"
        )

    def expand_query(self, original_query: str, contexts: List[Dict]) -> List[str]:
        """Generate context-aware query variations"""
        try:
            context_terms = self._extract_context_terms(contexts)
            expansions = self._generate_llm_expansions(original_query, context_terms)

            all_queries = list({original_query, *expansions})
            logger.info(f"Generated {len(expansions)} query variations")
            return all_queries

        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return [original_query]

    def _generate_llm_expansions(
        self, query: str, context_terms: Set[str], number: int = 4
    ) -> List[str]:
        """Generate context-aware query variations using Ollama"""
        query_expansion_prompt = f"""
          You are part of an information system that processes user queries.
          You expand a given query into {number} queries that incorporate relevant context terms.

          Structure:
          Follow the structure shown below in examples to generate expanded queries with context terms.
          Examples:
          1. Example Query 1: "climate change effects"
          Context Terms: ["temperature", "sea level", "weather"]
          Example Expanded Queries: ["temperature effects of climate change", "sea level rise due to global warming", "weather changes from environmental effects"]

          2. Example Query 2: "machine learning algorithms"
          Context Terms: ["classification", "regression", "optimization"]
          Example Expanded Queries: ["classification in machine learning", "regression techniques in algorithms", "optimization strategies for learning"]

          Your Task:
          Original Query: "{query}"
          Context Terms: {context_terms}
          Example Expanded Queries:
          """

        try:
            response = self.generator.run(query_expansion_prompt)
            return self._parse_llm_response(response["replies"][0])
        except Exception as e:
            logger.error(f"LLM expansion failed: {str(e)}")
            return []

    def _extract_context_terms(self, contexts: List[Dict]) -> Set[str]:
        """Extract significant terms from top contexts"""
        term_counter = Counter()
        for ctx in contexts:
            content = ctx.get("content", "")
            doc = self.nlp(content)
            terms = self._extract_significant_terms(doc)
            term_counter.update(terms)
        return {term for term, _ in term_counter.most_common(5)}

    def _extract_significant_terms(self, doc) -> List[str]:
        """Extract key terms using linguistic features"""
        terms = []
        # Named entities
        terms += [ent.text.lower() for ent in doc.ents]
        # Nouns and proper nouns
        terms += [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
        ]
        # Compound nouns
        terms += [chunk.text.lower() for chunk in doc.noun_chunks]
        return list(set(terms))

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response into individual queries"""
        queries = []
        for line in response.split("\n"):
            if "." in line:
                query = line.split(".", 1)[1].strip()
                if query:
                    queries.append(query)
        return queries


class HybridRetriever:
    def __init__(self):
        # Initialize document store
        self.document_store = InMemoryDocumentStore()
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True,
        )
        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=200,
            split_overlap=50,
            respect_sentence_boundary=True,
        )

        # Initialize embedder
        self.embedder = SentenceTransformersDocumentEmbedder(
            model="all-MiniLM-L6-v2", device=device
        )

        # Initialize retrievers
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)
        self.vector_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store
        )

        # Configure weights
        self.weights = {"bm25": 0.4, "vector": 0.6}

        # Warm up the embedding model
        self.warm_up()

    def warm_up(self):
        """Warm up the embedding model"""
        try:
            self.embedder.warm_up()
            logger.info("Embedding model warmed up successfully")
            self.splitter.warm_up()
            logger.info("Document splitter warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up embedding model: {str(e)}")
            raise

    def _initialize_store(self):
        """Initialize the document store with required parameters"""
        try:
            # Get all document IDs
            all_docs = self.document_store.filter_documents()
            if all_docs:
                doc_ids = [doc.id for doc in all_docs]
                # Delete documents if any exist
                self.document_store.delete_documents(document_ids=doc_ids)

            # Set up embedding dimension based on the model
            self.embedding_dim = 768  # Default for sentence-transformers
            self.document_store.embedding_dim = self.embedding_dim

        except Exception as e:
            logger.error(f"Error initializing document store: {str(e)}")

    # Update the HybridRetriever's index_documents method
    async def index_documents(self, documents: List[Dict]):
        """Index documents with proper embeddings"""
        try:
            # Convert to Haystack documents with embeddings
            haystack_docs = []
            for doc in documents:
                if doc.get("content"):
                    # Create Document with embedding
                    haystack_doc = Document(
                        content=doc["content"],
                        meta={
                            "url": doc["url"],
                            "title": doc.get("title", ""),
                            "doc_type": doc.get("doc_type", "text"),
                        },
                        embedding=doc.get(
                            "embedding"
                        ),  # Add embedding directly to Document
                    )
                    haystack_docs.append(haystack_doc)

            # Write documents to store (embeddings are part of Document objects)
            self.document_store.write_documents(haystack_docs)

        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")
            raise

    async def process_and_embed_documents(
        self, raw_documents: List[Dict]
    ) -> List[Document]:
        """Process documents and generate embeddings"""
        try:
            # Clean and create basic documents
            base_docs = [
                Document(
                    content=self._clean_content(doc["content"]),
                    meta={
                        "url": doc["url"],
                        "title": doc.get("title", ""),
                        "doc_type": doc.get("doc_type", "text"),
                    },
                )
                for doc in raw_documents
                if doc.get("content")
            ]

            # Clean documents
            cleaned_docs = self.cleaner.run(documents=base_docs)["documents"]

            # Split documents
            split_docs = self.splitter.run(documents=cleaned_docs)["documents"]

            # Generate embeddings
            embedding_result = self.embedder.run(documents=split_docs)
            return embedding_result["documents"]

        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return []

    async def retrieve(self, queries: List[str], top_k: int = 5) -> List[Document]:
        """Perform hybrid retrieval combining BM25 and dense retrieval"""
        try:
            all_docs = []

            for query in queries:
                # BM25 retrieval
                bm25_result = self.bm25_retriever.run(query=query, top_k=top_k)
                bm25_docs = bm25_result["documents"]
                print(f"BM25 retrieved {len(bm25_docs)} documents")

                # Generate query embedding
                query_embedding_result = self.embedder.run(
                    documents=[Document(content=query)]
                )
                query_embedding = query_embedding_result["documents"][0].embedding

                # Vector retrieval
                vector_result = self.vector_retriever.run(
                    query_embedding=query_embedding, top_k=top_k
                )
                vector_docs = vector_result["documents"]
                print(f"Vector retrieved {len(vector_docs)} documents")

                # Combine results
                combined_docs = self._combine_results(bm25_docs, vector_docs)
                all_docs.extend(combined_docs)

            # Deduplicate and return
            return self._deduplicate_docs(all_docs)

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []

    def _combine_results(
        self, bm25_docs: List[Document], vector_docs: List[Document]
    ) -> List[Document]:
        """Combine and score documents from both retrievers"""
        scored_docs = {}

        # Score BM25 results
        for idx, doc in enumerate(bm25_docs):
            score = (1 - idx / len(bm25_docs)) * self.weights["bm25"]
            doc_id = doc.id
            scored_docs[doc_id] = {"doc": doc, "score": score}

        # Score vector results
        for idx, doc in enumerate(vector_docs):
            score = (1 - idx / len(vector_docs)) * self.weights["vector"]
            doc_id = doc.id
            if doc_id in scored_docs:
                scored_docs[doc_id]["score"] += score
            else:
                scored_docs[doc_id] = {"doc": doc, "score": score}

        # Sort by combined score
        sorted_docs = sorted(
            scored_docs.values(), key=lambda x: x["score"], reverse=True
        )

        return [item["doc"] for item in sorted_docs]

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents while preserving order"""
        seen = set()
        unique_docs = []
        print(f"Original num docs: {len(docs)}")

        for doc in docs:
            if doc.id not in seen:
                seen.add(doc.id)
                unique_docs.append(doc)

        print(f"Deduplicated num docs: {len(unique_docs)}")
        return unique_docs

    def _clean_content(self, content: str) -> str:
        original_length = len(content)
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"[^\w\s\.,!?-]", "", content)
        content = content.strip()

        logger.debug(
            f"Content cleaned: {original_length} chars -> {len(content)} chars"
        )
        return content


class RetrieverAgent:
    def __init__(self, max_hops: int = 2):
        self.max_hops = max_hops
        self.searxng = SearxNGWrapper()
        self.query_expander = HybridQueryExpander()
        self.hybrid_retriever = HybridRetriever()
        self.supported_doc_types = {
            "pdf": self._process_pdf,
            "html": self._process_html,
            "doc": self._process_doc,
            "text": self._process_text,
        }
        self.circuit_breaker = CircuitBreaker(failure_threshold=10, reset_timeout=300)
        self.semaphore = Semaphore(5)
        self.cache = {}
        logger.info(f"RetrieverAgent initialized with max_hops={max_hops}")

    async def retrieve(self, query: str) -> List[Dict]:
        contexts = []
        seen_urls: Set[str] = set()
        current_queries = [query]

        for hop in range(self.max_hops):
            try:
                logger.info(f"\n{'='*40} Hop {hop+1} {'='*40}")
                # Get search results
                search_results = await self.searxng.search(current_queries[0])

                # Process and crawl search results
                processed_docs = await self._process_documents(
                    search_results, seen_urls
                )
                print(f"Processed {len(processed_docs)} documents in hop {hop+1}")
                print(f"Seen URLs: {len(seen_urls)}")
                print(f"Current contexts: {len(contexts)}")

                # Generate embeddings and index documents
                embedded_docs = await self.hybrid_retriever.process_and_embed_documents(
                    processed_docs
                )
                self.hybrid_retriever.document_store.write_documents(embedded_docs)

                # Generate expanded queries
                expanded_queries = []
                for q in current_queries:
                    expanded = self.query_expander.expand_query(q, contexts)
                    expanded_queries.extend(expanded)
                print(f"Expanded queries ({len(expanded_queries)}): {expanded_queries}")

                # Perform hybrid retrieval
                retrieved_docs = await self.hybrid_retriever.retrieve(expanded_queries)
                print(f"Retrieved {len(retrieved_docs)} documents in hop {hop+1}")

                print(f"Seen URLs: {len(seen_urls)}")

                # Process retrieved documents
                contexts = self.documents_to_json(retrieved_docs)
                print(f"Added {len(contexts)} new contexts in hop {hop+1}")
                for context in contexts:
                    print(
                        f"Title: {context['title']}\nURL: {context['url']}\nContent: {context['content']}\nRelevance: {context['relevance']}\nContext Length: {len(context['content'])}"
                    )

                if self._should_stop(contexts):
                    print(f"Stopping criteria met at hop {hop+1}")
                    break

                # Update queries for next hop
                current_queries = self._generate_next_queries(contexts)
                print(
                    f"Generated {len(current_queries)} queries for next hop: {current_queries}"
                )

            except Exception as e:
                logger.error(f"Error in hop {hop+1}: {str(e)}")
                self.circuit_breaker.record_failure()
                if self.circuit_breaker.is_open:
                    logger.warning("Circuit breaker open, using fallback retrieval")
                    return await self._fallback_retrieval(query)
                continue

        return self._post_process_contexts(contexts)

    def documents_to_json(self, docs: List[Document]) -> List[Dict]:
        """Convert a list of Document objects to a list of JSON-like dictionaries."""
        json_docs = []
        for doc in docs:
            json_doc = {
                "title": doc.meta.get("title", ""),
                "url": doc.meta["url"],
                "content": doc.content,
                "doc_type": doc.meta.get("doc_type", "text"),
                "relevance": getattr(doc, "score", 0.0),
            }
            json_docs.append(json_doc)
        return json_docs

    async def _process_documents(
        self, results: List[Dict], seen_urls: Set[str]
    ) -> List[Dict]:
        processed_docs = []
        async with asyncio.Semaphore(5):  # Limit concurrent processing
            tasks = [
                self._process_result_with_retry(result)
                for result in results
                if result["url"] not in seen_urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Document processing error: {str(result)}")
                    self.circuit_breaker.record_failure()
                elif result:
                    # Convert ProcessedDocument to dict if necessary
                    doc_dict = (
                        asdict(result)
                        if isinstance(result, ProcessedDocument)
                        else result
                    )
                    processed_docs.append(doc_dict)
                    seen_urls.add(doc_dict["url"])
        return processed_docs

    async def _process_result_with_retry(self, result: Dict) -> Optional[Dict]:
        if not self.circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open, skipping request")
            return None

        try:
            processed_doc = await self._process_result(result)
            # Convert ProcessedDocument to dict
            return (
                asdict(processed_doc)
                if isinstance(processed_doc, ProcessedDocument)
                else processed_doc
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500:
                logger.error(f"Server error for URL {result['url']} - skipping")
                return None
            raise
        except Exception as e:
            logger.error(f"Error processing {result['url']}: {str(e)}")
            self.circuit_breaker.record_failure()
            return None

    async def _fallback_retrieval(self, query: str) -> List[Dict]:
        logger.warning("Using fallback keyword-only retrieval")
        results = await self.searxng.search(query)
        return await self._process_documents(results, set())

    async def _process_result(self, result: Dict) -> Optional[ProcessedDocument]:
        try:
            url = result["url"]
            if not self._is_valid_url(url):
                return None

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8081/crawl",
                    json={"url": url},
                    headers={"User-Agent": "Mozilla/5.0"},  # Add headers
                )
                if response.status_code != 200:
                    return None

                content = response.json().get("content", "")

                if not content:
                    logger.warning(f"No content for URL {url}")
                    return None

                return ProcessedDocument(
                    title=result.get("title", url),
                    url=url,
                    content=content,
                    doc_type="markdown",
                    metadata={},
                    relevance_score=0.0,
                    processing_time=0.0,
                )

        except httpx.ReadTimeout:
            logger.warning(f"Timeout processing {url}")
            return None
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return None

    def _document_to_dict(self, doc: ProcessedDocument) -> Dict:
        return {
            "title": doc.title,
            "url": doc.url,
            "content": doc.content,
            "doc_type": doc.doc_type,
            "metadata": doc.metadata,
            "relevance": doc.relevance_score,
            "processing_time": doc.processing_time,
        }

    def _log_summary(
        self,
        total_time: float,
        contexts: List[Dict],
        seen_urls: Set[str],
        total_errors: int,
    ):
        logger.info(f"\nRetrieval completed in {total_time:.2f} seconds")
        logger.info(f"Total contexts gathered: {len(contexts)}")
        logger.info(f"Total unique URLs: {len(seen_urls)}")
        logger.info(f"Total errors: {total_errors}")

        print(f"\n{'='*50}")
        print("Retrieval Summary:")
        print(f"- Total time: {total_time:.2f} seconds")
        print(f"- Contexts found: {len(contexts)}")
        print(f"- Unique sources: {len(seen_urls)}")
        print(f"- Total errors: {total_errors}")
        print(f"{'='*50}\n")

    def _get_document_type(self, url: str) -> str:
        ext = urlparse(url).path.split(".")[-1].lower()
        doc_type = {
            "pdf": "pdf",
            "doc": "doc",
            "docx": "doc",
            "txt": "text",
            "html": "html",
            "htm": "html",
        }.get(ext, "html")

        logger.debug(f"Detected document type {doc_type} for URL: {url}")
        return doc_type

    async def _process_pdf(self, content: str, result: Dict) -> Dict:
        logger.info("Processing PDF document")
        return {
            "content": content,
            "metadata": {
                "type": "pdf",
                "pages": len(content.split("\f")),
                "structured": True,
            },
        }

    async def _process_html(self, content: str, result: Dict) -> Dict:
        logger.info("Processing HTML document")
        return {"content": content, "metadata": {"type": "html", "structured": False}}

    async def _process_doc(self, content: str, result: Dict) -> Dict:
        logger.info("Processing DOC document")
        return {"content": content, "metadata": {"type": "doc", "structured": True}}

    async def _process_text(self, content: str, result: Dict) -> Dict:
        logger.info("Processing plain text document")
        return {"content": content, "metadata": {"type": "text", "structured": False}}

    def _calculate_relevance(self, doc: Dict) -> float:
        score = 0.0
        content = doc.get("content", "")

        # Length score
        length_score = min(len(content) / 1000, 1.0) * 0.3
        score += length_score
        logger.debug(f"Length score: {length_score:.2f}")

        # Structure score
        structure_score = (
            0.2 if doc.get("metadata", {}).get("structured", False) else 0.0
        )
        score += structure_score
        logger.debug(f"Structure score: {structure_score:.2f}")

        # Content quality score
        quality_score = 0.3 if len(re.findall(r"\w+", content)) > 100 else 0.0
        score += quality_score
        logger.debug(f"Quality score: {quality_score:.2f}")

        final_score = min(score, 1.0)
        logger.debug(f"Final relevance score: {final_score:.2f}")
        return final_score

    def _generate_next_queries(self, contexts: List[Dict]) -> List[str]:
        """Generate queries for the next hop based on current contexts"""
        if not contexts:
            return []

        key_terms = self.query_expander._extract_context_terms(contexts)
        return [f"{term} related information" for term in key_terms]

    def _should_stop(self, contexts: List[Dict]) -> bool:
        if len(contexts) >= 15:
            logger.info("Stopping: Maximum context limit reached")
            return True

        quality_contexts = [ctx for ctx in contexts if ctx.get("relevance", 0) > 5]

        if len(quality_contexts) >= 5:
            logger.info("Stopping: Sufficient high-quality contexts found")
            return True

        return False

    def _post_process_contexts(self, contexts: List[Dict]) -> List[Dict]:
        logger.info("Post-processing contexts")
        processed = []

        for idx, ctx in enumerate(contexts, 1):
            logger.debug(f"Processing context {idx}")
            content = ctx.get("content", "")
            cleaned_content = self._clean_content(content)

            processed.append(
                {
                    "title": ctx.get("title", ""),
                    "url": ctx.get("url", ""),
                    "content": cleaned_content,
                    "doc_type": ctx.get("doc_type", "unknown"),
                    "metadata": ctx.get("metadata", {}),
                    "relevance": ctx.get("relevance", 0),
                }
            )

        logger.info(f"Post-processing completed. Processed {len(processed)} contexts")
        return processed

    def _clean_content(self, content: str) -> str:
        original_length = len(content)
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"[^\w\s\.,!?-]", "", content)
        content = content.strip()

        logger.debug(
            f"Content cleaned: {original_length} chars -> {len(content)} chars"
        )
        return content

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @lru_cache(maxsize=100)
    def _get_cached_result(self, url: str) -> Optional[ProcessedDocument]:
        return self.cache.get(url)
