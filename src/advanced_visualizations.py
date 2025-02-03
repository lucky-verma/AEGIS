import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
import plotly.figure_factory as ff
import networkx as nx
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


def create_visualizations(entities, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([e["text"] for e in entities])

    # 3D Scatter Plot
    # Dimensionality reduction
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    scatter_3d = go.Figure(
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

    scatter_3d.update_layout(
        title="3D Latent Space of Top Entities",
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        width=800,
        height=800,
        margin=dict(r=0, b=0, l=0, t=40),
    )

    # Network Graph
    similarity_matrix = np.inner(embeddings, embeddings)
    G = nx.Graph()
    for i in range(len(entities)):
        G.add_node(i, name=entities[i]["text"], count=entities[i]["count"])
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if similarity_matrix[i][j] > 0.5:  # Adjust threshold as needed
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=[G.nodes[node]["count"] * 2 for node in G.nodes()],
            color=[G.nodes[node]["count"] for node in G.nodes()],
            sizemode="area",
            sizeref=2.0
            * max([G.nodes[node]["count"] for node in G.nodes()])
            / (40.0**2),
            sizemin=4,
        ),
        text=[
            f"{G.nodes[node]['name']}<br>Count: {G.nodes[node]['count']}"
            for node in G.nodes()
        ],
        textposition="top center",
    )

    network_graph = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Entity Relationship Network",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Hierarchical Clustering Dendrogram
    distance_matrix = pdist(embeddings)
    linkage_matrix = linkage(distance_matrix, method="ward")

    dendro = ff.create_dendrogram(
        embeddings, labels=[e["text"] for e in entities], orientation="left"
    )
    dendro.update_layout(title="Hierarchical Clustering of Entities")

    return scatter_3d, network_graph, dendro
