import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap


def create_visualizations(data):
    visualizations = []

    try:
        # Semantic Space Projection
        if "contexts" in data and len(data["contexts"]) > 1:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [ctx["content"] for ctx in data["contexts"]]
            embeddings = model.encode(texts)

            reducer = umap.UMAP(n_components=3, random_state=42)
            projections = reducer.fit_transform(embeddings)

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=projections[:, 0],
                        y=projections[:, 1],
                        z=projections[:, 2],
                        mode="markers",
                        text=[ctx["title"] for ctx in data["contexts"]],
                        marker=dict(
                            size=5, color=projections[:, 2], colorscale="Viridis"
                        ),
                    )
                ]
            )
            fig.update_layout(title="Semantic Context Space")
            visualizations.append(fig)

        # Answer Quality Radar Chart
        if "evaluation" in data:
            eval_data = data["evaluation"]["scores"]
            fig = go.Figure(
                data=go.Scatterpolar(
                    r=[
                        eval_data.get("accuracy_score", 0) / 10,
                        eval_data.get("completeness_score", 0) / 10,
                        eval_data.get("relevance_score", 0) / 10,
                        eval_data.get("coherence_score", 0) / 10,
                        eval_data.get("factual_consistency_score", 0) / 10,
                    ],
                    theta=["Accuracy", "Completeness", "Relevance", "Coherence", "Factual Consistency"],
                    fill="toself",
                )
            )
            fig.update_layout(title="Answer Quality Assessment")
            visualizations.append(fig)

    except Exception as e:
        print(f"Visualization error: {str(e)}")

    return visualizations[:3]  # Return max 3 visualizations
