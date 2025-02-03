import pandas as pd
import streamlit as st
from search import perform_search
from entity_extraction import extract_entities, get_top_entities
from vector_store import store_entities, query_vector_store
from llm_processor import process_with_llm
from entity_visualization import create_3d_latent_space
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="Entity Search and Analysis", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
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

# Create two columns for input
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Enter an entity to search for:")

with col2:
    search_button = st.button("Search", key="search_button")

if search_button and query:
    with st.spinner("Searching and processing..."):
        search_results = perform_search(query)
        entities = extract_entities(search_results)
        store_entities(entities)
        relevant_info = query_vector_store(query)
        final_output = process_with_llm(query, relevant_info)

        # Get top 5 entities and create 3D visualization
        top_entities = get_top_entities(entities, n=5)
        fig_3d = create_3d_latent_space(top_entities)

    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(
        ["Structured Output", "3D Visualization", "Raw Search Results"]
    )

    with tab1:
        st.markdown("<div class='output-container'>", unsafe_allow_html=True)
        if final_output:
            st.subheader("Structured Output")
            st.write(final_output)
        else:
            st.write("No structured output available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_3d, use_container_width=True)
        with col2:
            st.subheader("Entity Statistics")
            stats_df = pd.DataFrame(top_entities)
            stats_df.columns = ['Entity', 'Type', 'Frequency']
            st.table(stats_df)


    with tab3:
        st.markdown("<div class='output-container'>", unsafe_allow_html=True)
        for result in search_results[:5]:
            st.subheader(result.get("title", "No Title"))
            st.write(result.get("url", "No URL"))
            st.write(result.get("content", "No Content"))
            st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.header("About")
st.sidebar.info(
    "This app performs entity search and analysis using web data and AI processing."
)
st.sidebar.markdown("---")
