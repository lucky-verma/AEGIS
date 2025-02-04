import time
import pandas as pd
import streamlit as st
from search import perform_search
from entity_extraction import extract_entities, get_top_entities
from vector_store import store_entities, query_vector_store
from llm_processor import process_with_llm
from advanced_visualizations import create_visualizations
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Entity Search and Analysis", layout="wide")

st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .output-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Entity Search and Analysis")

query = st.text_input("Enter an entity to search for:")
search_button = st.button("Search", key="search_button")


def process_query(query):
    steps = [
        ("Performing search", perform_search),
        ("Extracting entities", extract_entities),
        ("Storing entities", store_entities),
        ("Querying vector store", query_vector_store),
        ("Processing with LLM", process_with_llm),
        ("Getting top entities", lambda x: get_top_entities(x, n=10)),
        ("Creating visualizations", create_visualizations),
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = {}

    for i, (description, func) in enumerate(steps):
        progress = int((i / len(steps)) * 100)
        progress_bar.progress(progress)
        status_text.text(f"{description}... ({progress}%)")

        if i == 0:
            results["search_results"] = func(query)
        elif i == 1:
            results["entities"] = func(results["search_results"])
        elif i == 2:
            func(results["entities"])
        elif i == 3:
            results["relevant_info"] = func(query)
        elif i == 4:
            results["final_output"] = func(query, results["relevant_info"])
        elif i == 5:
            results["top_entities"] = func(results["entities"])
        elif i == 6:
            results["visualizations"] = func(results["top_entities"])

    progress_bar.progress(100)
    status_text.success("Processing complete!")
    progress_bar.empty()
    time.sleep(2)
    status_text.empty()
    return results


if search_button and query:
    results = process_query(query)

    tab1, tab2 = st.tabs(
        ["Structured Output", "3D Visualization"]
    )

    with tab1:
        st.markdown("<div class='output-container'>", unsafe_allow_html=True)
        st.write(
            results["final_output"]
            if results["final_output"]
            else "No structured output available."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(results["visualizations"][0], use_container_width=True)
            st.plotly_chart(results["visualizations"][1], use_container_width=True)
        with col2:
            st.subheader("Entity Statistics")
            stats_df = pd.DataFrame(results["top_entities"])
            stats_df.columns = ["Entity", "Type", "Frequency"]
            st.table(stats_df)
            st.plotly_chart(results["visualizations"][2], use_container_width=True)

    st.sidebar.header("Search Results")
    for result in results["search_results"]:
        st.sidebar.subheader(result.get("title", "No Title"))
        st.sidebar.markdown(
            f"[{result.get('url', 'No URL')}]({result.get('url', '#')})",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            result.get("content", "No Content")[:500] + "...",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown("---")

st.sidebar.header("About")
st.sidebar.info(
    "This app performs entity search and analysis using web data and AI processing."
)
st.sidebar.markdown("---")
