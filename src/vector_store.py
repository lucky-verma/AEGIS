from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(
    VECTOR_DIMENSION
)  # 384 is the dimensionality of the chosen model
entities = []  # Initialize an empty list to store entities


def store_entities(new_entities):
    global entities
    entities.extend(new_entities)
    texts = [f"{e['text']} ({e['label']})" for e in new_entities]
    embeddings = model.encode(texts)
    index.add(embeddings)


def query_vector_store(query):
    query_embedding = model.encode([query])
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    relevant_info = [
        {
            "text": entities[i]["text"],
            "label": entities[i]["label"],
            "source": entities[i]["source"],
        }
        for i in indices[0]
        if i < len(entities)
    ]
    return relevant_info
