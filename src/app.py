import json
import logging
import sys
import streamlit as st
import asyncio
from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.evaluator import EvaluatorAgent
from advanced_visualizations import create_visualizations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize agents
retriever = RetrieverAgent(max_hops=2)
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
        print("\n=== GENERATED REASONING ===")
        print(answer)
        print("====================\n")

        evaluation = await evaluator.evaluate(query, answer)
        print("\n=== EVALUATION RESULT ===")
        print(json.dumps(evaluation, indent=2))
        print("=====================\n")

        return {
            "query": query,
            "answer": answer,
            "evaluation": evaluation,
            "contexts": contexts,
            "visualizations": create_visualizations(
                {"contexts": contexts, "answer": answer, "evaluation": evaluation}
            ),
        }
    except Exception as e:
        print(f"\n=== ERROR IN PROCESS_QUERY ===\n{str(e)}\n========================\n")
        logger.exception(f"Error processing query: {str(e)}")
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
            st.markdown(ctx["content"][:100] + "...")
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
