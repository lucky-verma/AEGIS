import streamlit as st
import asyncio
from src.agents.retriever import RetrieverAgent
from src.agents.reasoner import ReasonerAgent
from src.agents.evaluator import EvaluatorAgent
from src.advanced_visualizations import create_visualizations

# Configuration
SCRAPER_SERVICE = st.secrets.get("SCRAPER_SERVICE", "http://host.docker.internal:8081")
SEARXNG_HOST = st.secrets.get("SEARXNG_HOST", "http://host.docker.internal:8080")
OLLAMA_HOST = st.secrets.get("OLLAMA_HOST", "http://host.docker.internal:11434")


def main():
    st.set_page_config(page_title="Agentic RAG System", layout="wide")
    st.title("Multi-Hop Entity Search System")

    query = st.text_input("Enter your complex query:")
    if st.button("Search"):
        with st.spinner("Processing with multi-hop reasoning..."):
            results = asyncio.run(process_query(query))
            display_results(results)


async def process_query(query: str):
    """Orchestrate the multi-hop RAG process"""
    try:
        retriever = RetrieverAgent(max_hops=3)
        reasoner = ReasonerAgent()
        evaluator = EvaluatorAgent()

        # Step 1: Multi-hop retrieval
        contexts = await retriever.retrieve(query)

        # Step 2: Final answer generation
        answer = await reasoner.reason(query, contexts)

        # Step 3: Answer evaluation
        evaluation = await evaluator.evaluate(query, answer)

        # Step 4: Create visualizations
        visualizations = create_visualizations(
            {
                "entities": contexts,
                "texts": [ctx["content"] for ctx in contexts],
                "answer": answer,
            }
        )

        return {
            "query": query,
            "answer": answer,
            "evaluation": evaluation,
            "contexts": contexts,
            "visualizations": visualizations,
        }

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return {}


def display_results(results):
    """Display results in interactive panels"""
    if not results:
        st.warning("No results found. Try refining your query.")
        return

    # Main answer section
    with st.expander("✨ Generated Answer", expanded=True):
        st.markdown(f"**Question:** {results['query']}")
        st.markdown(f"**Answer:** {results['answer']}")
        st.divider()
        st.json(results["evaluation"])

    # Context sources
    with st.expander("🔍 Source Contexts"):
        for idx, ctx in enumerate(results["contexts"], 1):
            st.subheader(f"Source {idx}: {ctx.get('title', 'Untitled')}")
            st.caption(f"[{ctx['url']}]({ctx['url']})")
            st.markdown("``````")
            st.divider()

    # Analytics visualizations
    with st.expander("📊 Analysis"):
        if results.get("visualizations"):
            cols = st.columns(2)
            for idx, fig in enumerate(results["visualizations"]):
                cols[idx % 2].plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualization data available")


if __name__ == "__main__":
    main()
