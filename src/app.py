import logging
import sys
import streamlit as st
import asyncio
from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.evaluator import EvaluatorAgent
from advanced_visualizations import create_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agentic RAG System", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; }
    .stProgress .st-bo { height: 20px; }
    .stAlert { padding: 10px; }
    </style>
""",
    unsafe_allow_html=True,
)


# Initialize agents
@st.cache_resource
def init_agents():
    return RetrieverAgent(max_hops=2), ReasonerAgent(), EvaluatorAgent()


retriever, reasoner, evaluator = init_agents()


def main():

    # Sidebar Configuration
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/your-repo/logo.png", width=100)
        st.title("Settings")

        # Retrieval Settings
        st.header("🔍 Retrieval Settings")
        retriever.max_hops = st.slider("Max Hops", 1, 5, 2)
        st.number_input("Results per Hop", 3, 10, 5)

        # Reasoning Settings
        st.header("🧠 Reasoning Settings")
        st.selectbox("Reasoning Mode", ["Detailed", "Concise"], index=0)

        # Evaluation Settings
        st.header("⚖️ Evaluation Settings")
        st.multiselect(
            "Evaluation Metrics",
            [
                "Accuracy",
                "Completeness",
                "Relevance",
                "Coherence",
                "Factual Consistency",
            ],
            ["Accuracy", "Completeness", "Relevance"],
        )

        st.markdown("---")
        st.info("💡 Adjust settings to customize the search and analysis process.")

    # Main Content
    st.title("🔮 Multi-Hop Entity Search System")
    st.markdown("### Discover deep connections with advanced reasoning")

    # Query Input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="Try: 'Who is John Smith and what are his contributions?'",
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)

    if search_button and query:
        process_and_display_results(query)


def process_and_display_results(query):
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with st.spinner():
            # Retrieval Phase
            status_text.text("🔍 Retrieving information...")
            progress_bar.progress(25)
            results = asyncio.run(process_query(query))

            if results:
                # Display Results
                progress_bar.progress(100)
                status_text.text("✨ Results ready!")
                display_enhanced_results(results)
            else:
                st.error("⚠️ No results found. Try refining your query.")
    except Exception as e:
        st.error(f"🚫 An error occurred: {str(e)}")
    finally:
        # Cleanup
        progress_bar.empty()
        status_text.empty()


def display_enhanced_results(results):
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Answer", "🔍 Sources", "📊 Analysis", "⚖️ Evaluation"]
    )

    with tab1:
        st.markdown("### Generated Answer")
        st.info(f"**Query:** {results['query']}")
        st.success(f"**Answer:** {results['answer']}")

        # Confidence Score
        confidence = results["evaluation"].get("confidence_score", 0)
        st.progress(confidence / 10)
        st.caption(f"Confidence Score: {confidence}/10")

    with tab2:
        st.markdown("### Source Documents")
        for idx, ctx in enumerate(results["contexts"], 1):
            with st.expander(f"Source {idx}: {ctx.get('title', 'Untitled')}"):
                st.caption(f"URL: {ctx['url']}")
                st.markdown(ctx["content"][:500] + "...")
                st.caption(f"Relevance Score: {ctx.get('relevance', 0):.2f}")

    with tab3:
        st.markdown("### Visual Analysis")
        if results.get("visualizations"):
            cols = st.columns(2)
            for idx, fig in enumerate(results["visualizations"]):
                cols[idx % 2].plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualization data available")

    with tab4:
        st.markdown("### Evaluation Details")
        col1, col2 = st.columns(2)

        with col1:
            st.json(results["evaluation"])

        with col2:
            if "detailed_feedback" in results["evaluation"]:
                feedback = results["evaluation"]["detailed_feedback"]

                st.markdown("#### 💪 Strengths")
                for strength in feedback.get("strengths", []):
                    st.success(strength)

                st.markdown("#### 🎯 Areas for Improvement")
                for weakness in feedback.get("weaknesses", []):
                    st.warning(weakness)

                st.markdown("#### 💡 Suggestions")
                for suggestion in feedback.get("suggestions", []):
                    st.info(suggestion)


async def process_query(query: str):
    try:
        contexts = await retriever.retrieve(query)
        answer = await reasoner.reason(query, contexts)
        evaluation = await evaluator.evaluate(query, answer)

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
        logger.exception(f"Error processing query: {str(e)}")
        return None


if __name__ == "__main__":
    main()
