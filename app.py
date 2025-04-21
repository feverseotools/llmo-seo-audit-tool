import streamlit as st
import time

# --- Set Page Config FIRST ---
# This MUST be the first Streamlit command executed
st.set_page_config(layout="wide")

# --- Import Modules & Check Availability ---
# Ensure the 'modules' directory is in the same path or Python path
try:
    from modules.content_processor import fetch_content_from_url
    # Import availability flags from analysis_engine
    from modules.analysis_engine import (
        analyze_content_chunking,
        analyze_entity_presence,
        analyze_semantic_intent,
        analyze_structured_data,
        analyze_llm_parsing,
        analyze_zero_click_signals,
        gemini_available, # Import flag
        openai_available  # Import flag
    )
    from modules.visualization import display_results
except ImportError as e:
    # Now it's safe to call st.error here
    st.error(f"Fatal Error: Could not import necessary modules: {e}. Please check dependencies and file structure.")
    st.stop() # Stop execution if modules can't be imported
except NameError as e:
    # Handle case where flags might not be defined if analysis_engine fails early
    st.error(f"Fatal Error: Could not import analysis flags: {e}. Check 'modules/analysis_engine.py'.")
    gemini_available = False # Assume unavailable if import fails
    openai_available = False
    # st.stop() # Decide if you want to stop or continue with limited functionality

# --- Streamlit App UI ---
st.title("🤖 LLMO Audit Assistant")

# --- API Key Input in Sidebar ---
st.sidebar.header("API Configuration")

# LLM Provider Selection - Disable option if library not available
provider_options = []
if gemini_available:
    provider_options.append("Gemini")
else:
    st.sidebar.warning("Gemini library not found. Install `google-generativeai`.")

if openai_available:
    provider_options.append("OpenAI")
else:
    st.sidebar.warning("OpenAI library not found. Install `openai>=1.0`.")

# Stop execution if no LLM providers are available at all
if not provider_options:
    st.sidebar.error("No LLM libraries (Gemini or OpenAI) found. Please install required libraries via requirements.txt.")
    st.stop()

# Determine default index safely
default_provider_index = 0
if "Gemini" in provider_options:
     default_provider_index = provider_options.index("Gemini")
elif "OpenAI" in provider_options: # Should be the only other option if list not empty
     default_provider_index = provider_options.index("OpenAI")


llm_provider = st.sidebar.radio(
    "Select LLM Provider:",
    provider_options, # Only show available options
    index=default_provider_index, # Set default selection
    key="llm_provider_select",
    help="Choose the LLM you want to use for analysis. Options disabled if library is missing."
)

api_key = None
library_ok_for_provider = False # Flag to check if selected provider's library is OK
if llm_provider == "Gemini":
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password", key="gemini_api_key_input_single", help="Enter your Google AI Gemini API key.")
    st.sidebar.caption("Using Google Gemini for analysis.")
    library_ok_for_provider = gemini_available # Check flag for Gemini
elif llm_provider == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input_single", help="Enter your OpenAI API key.")
    st.sidebar.caption("Using OpenAI (GPT) for analysis.")
    library_ok_for_provider = openai_available # Check flag for OpenAI

# --- Main Area ---
input_method = st.radio(
    "Choose Input Method:", ("URL", "Raw Content"), horizontal=True, key="input_method_radio_main",
    help="Select URL to analyze a webpage or Raw Content to paste text directly."
)
url_input = None
raw_content_input = None
if input_method == "URL":
    url_input = st.text_input("Enter URL:", placeholder="https://example.com", key="url_input_field_main", help="Enter the full URL.")
else:
    raw_content_input = st.text_area("Paste Raw Content:", height=250, key="raw_content_area_main", help="Paste the text content.")

# Analyze Button - Disable if key missing OR library for selected provider missing
key_provided = bool(api_key)
ready_to_analyze = key_provided and library_ok_for_provider # Combined check

analyze_button = st.button(
    "Analyze ✨", key="analyze_button_main", disabled=not ready_to_analyze,
    help=f"Click to start analysis using {llm_provider}. Requires API key and installed library."
)

# Show specific warnings if button clicked while disabled
if analyze_button and not ready_to_analyze:
    if not key_provided:
         st.error(f"🚨 Please enter the {llm_provider} API key in the sidebar!")
    elif not library_ok_for_provider:
         st.error(f"🚨 Cannot analyze using {llm_provider}: Required library is missing. Please check installation and restart.")

# --- Analysis Execution ---
if analyze_button and ready_to_analyze: # Use combined check
    final_results = {}
    content_valid = False
    processed_text = None
    soup_object = None

    # --- Validate Input and Fetch/Assign Content ---
    if input_method == "URL":
        if url_input:
            with st.spinner(f"Fetching and processing content from {url_input}..."):
                processed_text, soup_object = fetch_content_from_url(url_input)
            if processed_text is not None:
                st.success(f"Successfully fetched content (approx. {len(processed_text)} chars).")
                content_valid = True
        else:
            st.warning("Please enter a URL.")
    else: # Raw Content
        if raw_content_input and len(raw_content_input.strip()) > 10:
            processed_text = raw_content_input.strip()
            soup_object = None
            st.success(f"Using provided raw content (approx. {len(processed_text)} chars).")
            content_valid = True
        elif not raw_content_input or len(raw_content_input.strip()) <= 10:
             st.warning("Please paste sufficient content.")

    # --- Perform Analysis if Content is Valid ---
    if content_valid and processed_text is not None:
        st.info(f"Starting LLMO analysis modules using {llm_provider}...")
        analysis_progress = st.progress(0)
        status_text = st.empty()
        try:
            # Prepare config, including the selected provider and key
            analysis_config = {
                "provider": llm_provider.lower(), "api_key": api_key,
                "soup": soup_object, "input_url": url_input if input_method == "URL" else None
            }
            # Define analysis steps
            analysis_steps = [
                ("Content Chunking", analyze_content_chunking),
                ("Entity Presence", analyze_entity_presence),
                ("Semantic Intent", analyze_semantic_intent),
                ("Structured Data", analyze_structured_data),
                ("LLM Parsing", analyze_llm_parsing),
                ("Zero Click Signals", analyze_zero_click_signals),
            ]
            total_steps = len(analysis_steps)

            # Run analysis functions
            all_results_ok = True
            for i, (name, func) in enumerate(analysis_steps):
                status_text.text(f"Running: {name}...")
                # Call function with processed_text and the config dictionary
                result = func(processed_text, analysis_config)
                final_results[name.replace(" ", "_")] = result
                # Check if the result indicates a critical failure (e.g., missing library handled inside engine)
                if isinstance(result, dict) and result.get("recommendations"):
                    if any(rec.get("priority") == "Critical" for rec in result["recommendations"]):
                        all_results_ok = False
                        st.error(f"Critical error encountered during '{name}' analysis. See details below.")
                        # Optionally break the loop if one critical error should stop all analysis
                        # break
                analysis_progress.progress((i + 1) / total_steps)

            # Display final status
            if all_results_ok:
                status_text.success("Analysis Complete!")
            else:
                status_text.error("Analysis finished with critical errors.")
            time.sleep(1.5) # Keep message visible briefly
            status_text.empty(); analysis_progress.empty()

            # Display results even if there were errors (errors shown in recommendations)
            display_results(final_results)

            # Optional: Display analyzed content
            with st.expander("View Analyzed Content Snippet"):
                 display_text = str(processed_text)
                 st.text(display_text[:1500] + "..." if len(display_text) > 1500 else display_text)

        except Exception as e:
            st.error(f"An unexpected error occurred during the analysis orchestration: {e}")
            status_text.empty(); analysis_progress.empty()

    elif analyze_button and not content_valid:
        st.warning("Analysis could not proceed: No valid content provided or fetched.")
