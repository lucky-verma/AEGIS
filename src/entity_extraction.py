import spacy
from collections import Counter
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load a more comprehensive spaCy model for better entity recognition
nlp = spacy.load("en_core_web_trf")


def preprocess_text(text: str) -> str:
    """
    Preprocess the text by removing special characters and lowercasing.
    """
    return " ".join(token.text.lower() for token in nlp(text) if token.is_alpha)


def extract_entities(search_results: List[Dict]) -> List[Dict]:
    entities = []
    for result in search_results:
        # Combine title and content for better context
        full_text = f"{result['searxng']['title']} {result['searxng']['content']} {result['crawl4ai']}"
        doc = nlp(full_text)
        for ent in doc.ents:
            if ent.label_ in [
                "PERSON",
                "ORG",
                "GPE",
                "DATE",
                "SKILL",
            ]:  # Focus on relevant entity types
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "source": result["searxng"]["url"],
                    }
                )
    return entities


def get_top_entities(entities: List[Dict], n: int = 5) -> List[Dict]:
    entity_counter = Counter([(e["text"].lower(), e["label"]) for e in entities])
    top_entities = [
        {"text": e[0], "label": e[1], "count": count}
        for (e, count) in entity_counter.most_common(n)
    ]
    return top_entities


def rank_results(query: str, search_results: List[Dict]) -> List[Dict]:
    """
    Rank search results based on relevance to the query using TF-IDF and cosine similarity.
    """
    vectorizer = TfidfVectorizer()
    corpus = [preprocess_text(result["crawl4ai"]) for result in search_results]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    return [search_results[i] for i in ranked_indices]


def extract_skills(doc: spacy.tokens.Doc) -> List[str]:
    """
    Extract potential skills from the document.
    """
    skills = []
    for ent in doc.ents:
        if ent.label_ == "SKILL" or (
            ent.label_ == "ORG" and len(ent.text.split()) <= 3
        ):
            skills.append(ent.text)
    return list(set(skills))


def summarize_profile(entities: List[Dict], search_results: List[Dict]) -> Dict:
    """
    Create a summary of the profile based on extracted entities and search results.
    """
    name = next((e["text"] for e in entities if e["label"] == "PERSON"), "Unknown")
    organizations = list(set(e["text"] for e in entities if e["label"] == "ORG"))
    locations = list(set(e["text"] for e in entities if e["label"] == "GPE"))

    full_text = " ".join([result["crawl4ai"] for result in search_results])
    doc = nlp(full_text)
    skills = extract_skills(doc)

    return {
        "name": name,
        "current_organization": organizations[0] if organizations else "Unknown",
        "locations": locations,
        "skills": skills[:5],  # Top 5 skills
        "organizations": organizations,
    }


def process_search_results(query: str, search_results: List[Dict]) -> Dict:
    """
    Process search results to extract and summarize relevant information.
    """
    ranked_results = rank_results(query, search_results)
    entities = extract_entities(ranked_results)
    top_entities = get_top_entities(entities, n=10)
    profile_summary = summarize_profile(entities, ranked_results)

    return {
        "profile_summary": profile_summary,
        "top_entities": top_entities,
        "ranked_results": ranked_results[:3],  # Top 3 most relevant results
    }
