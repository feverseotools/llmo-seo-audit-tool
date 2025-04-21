import streamlit as st
import time  # To simulate analysis time

# --- Set Page Config FIRST ---
# This MUST be the first Streamlit command executed
st.set_page_config(layout="wide")

# --- Import Modules (with error handling) ---
# Ensure the 'modules' directory is in the same path or Python path
try:
    from modules.content_processor import fetch_content_from_url
    from modules.analysis_engine import (
        analyze_content_chunking,
        analyze_entity_presence,
        analyze_semantic_intent,
        analyze_structured_data,
        analyze_llm_parsing,
        analyze_zero_click_signals
        # Import any new analysis functions if you create them
    )
    from modules.visualization import display_results
except ImportError as e:
    # Now it's safe to call st.error here
    st.error(f"Fatal Error: Could not import necessary modules: {e}. Please check dependencies and file structure.")
    st.stop() # Stop execution if modules can't be imported

# --- Streamlit App UI ---
st.title("🤖 LLMO Audit Assistant")

# --- API Key Input in Sidebar ---
st.sidebar.header("API Configuration")
llm_provider = st.sidebar.radio(
    "Select LLM Provider:",
    ("Gemini", "OpenAI"),
    key="llm_provider_select",
    help="Choose the LLM you want to use for analysis."
)
api_key = None
if llm_provider == "Gemini":
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password", key="gemini_api_key_input_single", help="Enter your Google AI Gemini API key.")
    st.sidebar.caption("Using Google Gemini for analysis.")
elif llm_provider == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input_single", help="Enter your OpenAI API key.")
    st.sidebar.caption("Using OpenAI (GPT) for analysis.")

# --- Main Area ---
input_method = st.radio(
    "Choose Input Method:", ("URL", "Raw Content"), horizontal=True, key="input_method_radio_main",
    help="Select URL to analyze a webpage or Raw Content to paste text directly."
)
url_input = None
raw_content_input = None
if input_method == "URL":
    url_input = st.text_input("Enter URL:", placeholder="https://example.com", key="url_input_field_main", help="Enter the full URL (including http:// or https://).")
else:
    raw_content_input = st.text_area("Paste Raw Content:", height=250, key="raw_content_area_main", help="Paste the text content you want to analyze here.")

key_provided = bool(api_key)
analyze_button = st.button(
    "Analyze ✨", key="analyze_button_main", disabled=not key_provided,
    help=f"Click to start analysis using {llm_provider}. Requires the {llm_provider} API key in the sidebar."
)

if analyze_button and not key_provided:
     st.error(f"🚨 Please enter the {llm_provider} API key in the sidebar to proceed!")

# --- Analysis Execution ---
if analyze_button and key_provided:
    final_results = {}
    content_valid = False
    processed_text = None
    soup_object = None # Variable to hold the soup object for URL inputs

    # --- Validate Input and Fetch/Assign Content ---
    if input_method == "URL":
        if url_input:
            with st.spinner(f"Fetching and processing content from {url_input}..."):
                processed_text, soup_object = fetch_content_from_url(url_input)
            if processed_text is not None:
                st.success(f"Successfully fetched and processed content (approx. {len(processed_text)} characters).")
                content_valid = True
        else:
            st.warning("Please enter a URL to analyze.")
    else: # Raw Content
        if raw_content_input and len(raw_content_input.strip()) > 10:
            processed_text = raw_content_input.strip()
            soup_object = None
            st.success(f"Using provided raw content (approx. {len(processed_text)} characters).")
            content_valid = True
        elif not raw_content_input or len(raw_content_input.strip()) <= 10:
             st.warning("Please paste sufficient content (more than 10 characters) to analyze.")

    # --- Perform Analysis if Content is Valid ---
    if content_valid and processed_text is not None:
        st.info(f"Starting LLMO analysis modules using {llm_provider}...")
        analysis_progress = st.progress(0)
        status_text = st.empty()

        try:
            analysis_config = {
                "provider": llm_provider.lower(),
                "api_key": api_key,
                "soup": soup_object,
                "input_url": url_input if input_method == "URL" else None
            }
            analysis_steps = [
                ("Content Chunking", analyze_content_chunking),
                ("Entity Presence", analyze_entity_presence),
                ("Semantic Intent", analyze_semantic_intent),
                ("Structured Data", analyze_structured_data),
                ("LLM Parsing", analyze_llm_parsing),
                ("Zero Click Signals", analyze_zero_click_signals),
            ]
            total_steps = len(analysis_steps)

            for i, (name, func) in enumerate(analysis_steps):
                status_text.text(f"Running: {name}...")
                current_url = url_input if input_method == "URL" else None # Define current_url here
                # Pass config consistently
                final_results[name.replace(" ", "_")] = func(processed_text, analysis_config)
                analysis_progress.progress((i + 1) / total_steps)

            status_text.text("Analysis Complete!")
            time.sleep(1)
            status_text.empty()
            analysis_progress.empty()

            display_results(final_results)

            with st.expander("View Analyzed Content Snippet"):
                 display_text = str(processed_text)
                 st.text(display_text[:1500] + "..." if len(display_text) > 1500 else display_text)

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            status_text.empty()
            analysis_progress.empty()

    elif analyze_button and not content_valid:
        st.warning("Analysis could not proceed: No valid content provided or fetched.")
