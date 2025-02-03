import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def extract_entities(search_results):
    entities = []
    for result in search_results:
        doc = nlp(result["content"])
        for ent in doc.ents:
            entities.append(
                {"text": ent.text, "label": ent.label_, "source": result["url"]}
            )
    return entities


def get_top_entities(entities, n=5):
    entity_counter = Counter([(e["text"], e["label"]) for e in entities])
    top_entities = [
        {"text": e[0], "label": e[1], "count": count}
        for (e, count) in entity_counter.most_common(n)
    ]
    return top_entities
