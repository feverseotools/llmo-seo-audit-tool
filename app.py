import streamlit as st
import time  # To simulate analysis time

# Import functions from modules
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
    )
    from modules.visualization import display_results
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure the 'modules' directory and its files exist.")
    st.stop() # Stop execution if modules can't be imported

# --- Streamlit App UI ---

st.set_page_config(layout="wide") # Use wide layout
st.title("🤖 LLMO Audit Assistant")

# --- API Key Input in Sidebar ---
st.sidebar.header("API Configuration")
st.sidebar.info("Enter your API keys below to enable analysis. Keys are used for this session only and are not stored.")
# Use unique keys for widgets to prevent state issues if the script reruns unexpectedly
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input_main")
gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", key="gemini_api_key_input_main")

# --- Main Area ---
# Input Selection
input_method = st.radio(
    "Choose Input Method:",
    ("URL", "Raw Content"),
    horizontal=True,
    key="input_method_radio_main",
    help="Select URL to analyze a webpage or Raw Content to paste text directly."
)

# Input Fields
content_to_analyze = None
url_input = None
raw_content_input = None # Define raw_content_input here

if input_method == "URL":
    url_input = st.text_input(
        "Enter URL:",
        placeholder="https://example.com",
        key="url_input_field_main",
        help="Enter the full URL (including http:// or https://) of the page you want to analyze."
    )
else:
    raw_content_input = st.text_area(
        "Paste Raw Content:",
        height=250, # Slightly taller text area
        key="raw_content_area_main",
        help="Paste the text content you want to analyze here."
    )

# Analyze Button - Disable button if keys are missing
keys_provided = bool(openai_api_key and gemini_api_key)
analyze_button = st.button(
    "Analyze ✨",
    key="analyze_button_main",
    disabled=not keys_provided, # Disable button if keys are missing
    help="Click to start the analysis. Requires API keys to be entered in the sidebar."
)

# Show a warning if trying to analyze without keys
if analyze_button and not keys_provided:
     st.error("🚨 Please enter both OpenAI and Gemini API keys in the sidebar to proceed!")

# --- Analysis Execution ---

# Proceed only if button is clicked AND keys are provided
if analyze_button and keys_provided:
    final_results = {}
    content_valid = False
    processed_content = None # Use a different variable for the processed content

    # --- Validate Input and Fetch/Assign Content ---
    if input_method == "URL":
        if url_input:
            with st.spinner(f"Fetching and processing content from {url_input}..."):
                processed_content = fetch_content_from_url(url_input)
            if processed_content:
                st.success(f"Successfully fetched and processed content (approx. {len(processed_content)} characters).")
                content_valid = True
            else:
                # Error messages should be handled within fetch_content_from_url
                # st.error("Failed to fetch or extract content from URL.") # Keep commented unless fetch_content doesn't show errors
                pass # Error is shown by the function itself
        else:
            st.warning("Please enter a URL to analyze.")
    else: # Raw Content
        if raw_content_input and len(raw_content_input.strip()) > 10: # Add minimum length check
            processed_content = raw_content_input.strip()
            st.success(f"Using provided raw content (approx. {len(processed_content)} characters).")
            content_valid = True
        elif not raw_content_input or len(raw_content_input.strip()) <= 10:
             st.warning("Please paste sufficient content (more than 10 characters) to analyze.")


    # --- Perform Analysis if Content is Valid ---
    if content_valid and processed_content:
        st.info("Starting LLMO analysis modules...")
        analysis_progress = st.progress(0)
        status_text = st.empty()

        try:
            # Pass API keys to analysis functions
            api_keys = {"openai": openai_api_key, "gemini": gemini_api_key}

            # Define analysis steps for progress tracking
            analysis_steps = [
                ("Content Chunking", analyze_content_chunking),
                ("Entity Presence", analyze_entity_presence),
                ("Semantic Intent", analyze_semantic_intent),
                ("Structured Data", analyze_structured_data),
                ("LLM Parsing", analyze_llm_parsing),
                ("Zero Click Signals", analyze_zero_click_signals),
            ]
            total_steps = len(analysis_steps)

            # Run analysis functions sequentially and update progress
            for i, (name, func) in enumerate(analysis_steps):
                status_text.text(f"Running: {name}...")
                # Special handling for structured data needing URL
                if name == "Structured Data":
                     final_results[name.replace(" ", "_")] = func(processed_content, api_keys, url=url_input if input_method == "URL" else None)
                else:
                     final_results[name.replace(" ", "_")] = func(processed_content, api_keys)
                analysis_progress.progress((i + 1) / total_steps)

            status_text.text("Analysis Complete!")
            time.sleep(1) # Keep "Complete" message visible briefly
            status_text.empty() # Clear status text
            analysis_progress.empty() # Clear progress bar

            # Display the results using the imported visualization function
            display_results(final_results)

            # Optional: Display the content analyzed
            with st.expander("View Analyzed Content Snippet"):
                 # Ensure processed_content is a string before slicing
                 display_text = str(processed_content) if processed_content else ""
                 st.text(display_text[:1500] + "..." if len(display_text) > 1500 else display_text) # Show a bit more content

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            # Optionally clear progress/status if error occurs
            status_text.empty()
            analysis_progress.empty()

    # Handle cases where button was clicked, keys were present, but content wasn't valid
    elif analyze_button and not content_valid:
        st.warning("Analysis could not proceed: No valid content provided or fetched.")
