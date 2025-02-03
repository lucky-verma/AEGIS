from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go


def create_3d_latent_space(entities, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([e["text"] for e in entities])

    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                z=reduced_embeddings[:, 2],
                mode="markers+text",
                text=[
                    f"{e['text']} ({e['label']})<br>Count: {e['count']}"
                    for e in entities
                ],
                hoverinfo="text",
                marker=dict(
                    size=[
                        e["count"] * 2 for e in entities
                    ],  # Adjust size based on count
                    color=[e["count"] for e in entities],
                    colorscale="Viridis",
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.update_layout(
        title="3D Latent Space of Top Entities",
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        width=800,
        height=800,
        margin=dict(r=0, b=0, l=0, t=40),
    )

    return fig
