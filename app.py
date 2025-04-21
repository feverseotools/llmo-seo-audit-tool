import streamlit as st
import time  # To simulate analysis time

# Import functions from modules
from modules.content_processor import fetch_content_from_url
from modules.analysis_engine import (
    analyze_content_chunking,
    analyze_entity_presence,
    analyze_semantic_intent,
    analyze_structured_data,
    analyze_llm_parsing,
    analyze_zero_click_signals
)
from modules.visualization import display_results

# --- Streamlit App UI ---

st.set_page_config(layout="wide") # Use wide layout
st.title("🤖 LLMO Audit Assistant")

# Input Selection
input_method = st.radio("Choose Input Method:", ("URL", "Raw Content"), horizontal=True)

# Input Fields
content_to_analyze = None
url_input = None
raw_content_input = None # Define raw_content_input here

if input_method == "URL":
    url_input = st.text_input("Enter URL:", placeholder="https://example.com")
else:
    raw_content_input = st.text_area("Paste Raw Content:", height=200)

# Analyze Button
analyze_button = st.button("Analyze ✨")

# --- Analysis Execution ---

if analyze_button:
    final_results = {}
    content_valid = False
    processed_content = None # Use a different variable for the processed content

    if input_method == "URL":
        if url_input:
            with st.spinner(f"Fetching content from {url_input}..."):
                # Use the imported function
                processed_content = fetch_content_from_url(url_input)
            if processed_content:
                st.info(f"Successfully fetched content (approx. {len(processed_content)} characters).")
                content_valid = True
            else:
                # Error messages are handled within fetch_content_from_url now
                st.error("Failed to fetch or extract content from URL.")
        else:
            st.warning("Please enter a URL.")
    else: # Raw Content
        if raw_content_input:
            processed_content = raw_content_input # Assign content directly
            st.info(f"Using provided raw content (approx. {len(processed_content)} characters).")
            content_valid = True
        else:
            st.warning("Please paste some content.")

    # Proceed with analysis if content is valid
    if content_valid and processed_content:
        with st.spinner("Performing LLMO analysis... This may take a moment."):
            # Run all analysis functions using the imported engine functions
            # In a real app, you might run these in parallel if possible
            final_results["Content_Chunking"] = analyze_content_chunking(processed_content)
            final_results["Entity_Presence"] = analyze_entity_presence(processed_content)
            final_results["Semantic_Intent"] = analyze_semantic_intent(processed_content)
            final_results["Structured_Data"] = analyze_structured_data(processed_content, url=url_input if input_method == "URL" else None)
            final_results["LLM_Parsing"] = analyze_llm_parsing(processed_content)
            final_results["Zero_Click_Signals"] = analyze_zero_click_signals(processed_content)

        # Display the results using the imported visualization function
        display_results(final_results)

        # Optional: Display the content analyzed
        with st.expander("View Analyzed Content Snippet"):
             st.text(processed_content[:1000] + "..." if len(processed_content) > 1000 else processed_content)

    elif analyze_button and not content_valid:
        # If button clicked but no valid content, do nothing further
        pass
