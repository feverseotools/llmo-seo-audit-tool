import streamlit as st

# --- Helper Functions ---

def _get_score_color(score):
    """Returns color name based on score (internal helper)."""
    if not isinstance(score, (int, float)):
       return "grey" # Handle non-numeric scores gracefully
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange" # Using orange for medium scores
    else:
        return "red"

def _get_priority_color(priority):
    """Returns color name based on priority level (internal helper)."""
    priority_map = {
        "Critical": "#D32F2F",  # Darker Red
        "High": "#F44336",      # Red
        "Medium": "#FFA000",    # Orange
        "Low": "#4CAF50",       # Green
        "Info": "#1976D2"       # Blue for informational
    }
    # Return a default grey color if priority not in map
    return priority_map.get(priority, "#757575") # Grey


# --- Main Display Function ---

def display_results(results):
    """Displays the analysis results in the Streamlit app.

    Args:
        results (dict): A dictionary where keys are analysis types
                        and values are dicts containing 'score' and 'recommendations'.
    """
    if not results:
        st.warning("Analysis results are empty.")
        return

    st.subheader("📊 LLMO Audit Scores")
    st.markdown("---") # Divider

    # --- Score Display ---
    # Determine number of columns dynamically based on number of results
    num_results = len(results)
    num_cols = min(num_results, 3) # Max 3 columns, adjust as needed
    cols = st.columns(num_cols)
    col_idx = 0

    score_sections = {} # To hold data for recommendations section

    # Sort results alphabetically by key for consistent order
    sorted_result_keys = sorted(results.keys())

    for key in sorted_result_keys:
        data = results[key]
        if not isinstance(data, dict):
            st.error(f"Invalid data format for '{key}'. Expected a dictionary.")
            continue # Skip this result if data format is wrong

        score = data.get("score", 0) # Default to 0 if score is missing
        # Ensure score is numeric before proceeding
        if not isinstance(score, (int, float)):
            score = 0
            st.warning(f"Invalid score type for '{key}'. Using 0.")

        score = max(0, min(100, int(score))) # Clamp score between 0 and 100

        # Use columns for layout
        with cols[col_idx % num_cols]:
            # Use a container for better spacing and potential borders
            with st.container():
                 st.metric(label=key.replace("_", " ").title(), value=f"{score}/100")
                 # Custom progress bar using HTML for color control based on score
                 color = _get_score_color(score)
                 st.markdown(f"""
                 <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                     <div style="background-color: {color}; width: {score}%; height: 100%; border-radius: 5px;"></div>
                 </div>
                 """, unsafe_allow_html=True)
                 st.markdown("<br>", unsafe_allow_html=True) # Add some space below progress bar

            # Store data for recommendations
            score_sections[key] = data
        col_idx += 1

    st.markdown("---") # Visual separator

    # --- Recommendations Display ---
    st.subheader("💡 Recommendations")
    if not score_sections:
         st.info("No analysis sections found to display recommendations.")
         return

    for key in sorted_result_keys: # Use sorted keys here too
        data = score_sections.get(key)
        if not data or not isinstance(data, dict):
            continue # Skip if data is missing or invalid

        recommendations = data.get("recommendations", [])
        title = key.replace('_', ' ').title()

        with st.expander(f"{title} ({len(recommendations)} Recommendations)"):
            if recommendations and isinstance(recommendations, list):
                # Sort recommendations by priority (example: Critical > High > Medium > Low)
                priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
                try:
                    # Sort safely, handling missing 'priority' key
                    sorted_recs = sorted(
                        recommendations,
                        key=lambda x: priority_order.get(x.get('priority', 'Low'), 99) # Default to low priority if key missing
                    )
                except TypeError:
                     st.warning(f"Could not sort recommendations for {title} due to data format issues.")
                     sorted_recs = recommendations # Display unsorted if error

                for rec in sorted_recs:
                    if isinstance(rec, dict):
                        priority = rec.get("priority", "Info") # Default to Info if missing
                        color = _get_priority_color(priority)
                        text = rec.get('text', 'N/A')
                        # Use markdown for colored text and bold priority
                        st.markdown(f"<span style='color:{color}; font-weight:bold;'>[{priority}]</span> &nbsp; {text}", unsafe_allow_html=True)
                    else:
                        # Handle cases where a recommendation isn't a dictionary
                        st.markdown(f"<span style='color:grey;'>[Info]</span> &nbsp; {str(rec)}", unsafe_allow_html=True)

            elif not recommendations:
                st.write("✅ No specific recommendations for this section.")
            else:
                 st.warning("Recommendations data is not in the expected list format.")

