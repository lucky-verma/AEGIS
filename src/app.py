import streamlit as st
import asyncio
from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.evaluator import EvaluatorAgent
from advanced_visualizations import create_visualizations

# Initialize agents
retriever = RetrieverAgent(max_hops=3)
reasoner = ReasonerAgent()
evaluator = EvaluatorAgent()


def main():
    st.set_page_config(page_title="Agentic RAG System", layout="wide")
    st.title("Multi-Hop Entity Search System")

    query = st.text_input("Enter your complex query:")
    if st.button("Search"):
        with st.spinner("Processing with multi-hop reasoning..."):
            results = asyncio.run(process_query(query))
            display_results(results)


async def process_query(query: str):
    try:
        contexts = await retriever.retrieve(query)
        answer = await reasoner.reason(query, contexts)
        evaluation = await evaluator.evaluate(query, answer)
        visualizations = create_visualizations({"contexts": contexts, "answer": answer})
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
    if not results:
        st.warning("No results found. Try refining your query.")
        return

    with st.expander("✨ Generated Answer", expanded=True):
        st.markdown(f"**Question:** {results['query']}")
        st.markdown(f"**Answer:** {results['answer']}")
        st.divider()
        st.json(results["evaluation"])

    with st.expander("🔍 Source Contexts"):
        for idx, ctx in enumerate(results["contexts"], 1):
            st.subheader(f"Source {idx}: {ctx.get('title', 'Untitled')}")
            st.caption(f"[{ctx['url']}]({ctx['url']})")
            st.markdown(ctx["content"][:500] + "...")
            st.divider()

    with st.expander("📊 Analysis"):
        if results.get("visualizations"):
            cols = st.columns(2)
            for idx, fig in enumerate(results["visualizations"]):
                cols[idx % 2].plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualization data available")


if __name__ == "__main__":
    main()
