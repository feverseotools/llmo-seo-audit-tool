import time
import random
import streamlit as st
import re
import json # For parsing LLM responses
from collections import Counter
from urllib.parse import urlparse, urljoin

# --- Import libraries for Free Checks ---
# Using try-except blocks to handle potential import errors gracefully
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    st.error("NLTK library not found. Please install it (`pip install nltk`) and download necessary data.")
    nltk = None # Set to None to check later

try:
    from bs4 import BeautifulSoup # To handle soup object passed in config
except ImportError:
     st.error("BeautifulSoup4 library not found. Please install it (`pip install beautifulsoup4`).")
     BeautifulSoup = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    st.error("vaderSentiment library not found. Please install it (`pip install vaderSentiment`).")
    SentimentIntensityAnalyzer = None

# --- Import LLM Libraries & Set Availability Flags ---
gemini_available = False
try:
    import google.generativeai as genai
    # Perform a basic check if needed, e.g., accessing a constant
    if hasattr(genai, 'GenerativeModel'):
         gemini_available = True
         print("Google GenAI library loaded successfully.")
except ImportError:
    print("Google GenAI library not found.")
    genai = None # Ensure genai is None if import fails

openai_available = False
try:
    import openai
    # Check for modern client structure (version >= 1.0)
    if hasattr(openai, 'OpenAI'):
        openai_available = True
        print("OpenAI library (>=1.0) loaded successfully.")
    else:
        print("Older or unexpected OpenAI library structure found.")
except ImportError:
    print("OpenAI library not found.")
    openai = None # Ensure openai is None if import fails

# --- Download NLTK data (if library is available and data not found) ---
nltk_data_downloaded = True
if nltk:
    try:
        # Check for required data packages
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('sentiment/vader_lexicon.zip') # VADER lexicon often bundled here
    except LookupError as e:
        # Attempt to download missing package
        missing_data = str(e).split("'")[1] # Extract missing package name
        st.info(f"Downloading NLTK '{missing_data}' data...")
        try:
            nltk.download(missing_data, quiet=True)
            # Re-check after download attempt (optional, but good practice)
            nltk.data.find(f'tokenizers/{missing_data}' if 'punkt' in missing_data else f'corpora/{missing_data}' if 'stopwords' in missing_data else f'sentiment/{missing_data}')
        except Exception as download_error:
             st.error(f"Failed to download NLTK data '{missing_data}': {download_error}")
             nltk_data_downloaded = False # Flag that setup might be incomplete
    except Exception as general_nltk_error:
         # Catch other errors during NLTK data check
         st.error(f"An error occurred checking NLTK data: {general_nltk_error}")
         nltk_data_downloaded = False


# --- Initialize VADER (if library is available) ---
analyzer = None
if SentimentIntensityAnalyzer:
    try:
        analyzer = SentimentIntensityAnalyzer()
        print("VADER Sentiment Analyzer initialized.")
    except LookupError:
        # This might happen if the NLTK download above didn't cover it or failed
        st.warning("VADER lexicon might be missing. Ensure 'nltk.download(\"vader_lexicon\")' has run successfully.")
    except Exception as vader_init_error:
        st.error(f"Failed to initialize VADER Sentiment Analyzer: {vader_init_error}")


# --- Central LLM API Call Function ---
def _call_llm_api(prompt_template, content, analysis_config):
    """
    Calls the selected LLM API and returns the parsed result, checking library availability first.
    """
    provider = analysis_config.get("provider")
    api_key = analysis_config.get("api_key")
    model_name = provider.capitalize() if provider else "Unknown"

    print(f"--- Attempting LLM Call ({model_name}) ---")

    if not api_key or not provider:
         error_msg = f"Configuration Error: Missing API key or provider selection."
         print(error_msg)
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}

    # --- Check Library Availability ---
    if provider == 'gemini':
        if not gemini_available:
            error_msg = "Gemini provider selected, but 'google-generativeai' library is not installed or failed to load. Please check requirements.txt."
            st.error(error_msg) # Show error in UI
            return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
        if not genai: # Double check the module itself
             error_msg = "Internal Error: Gemini library loaded flag is true, but module is None."
             print(error_msg)
             return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    elif provider == 'openai':
        if not openai_available:
            error_msg = "OpenAI provider selected, but 'openai>=1.0' library is not installed or failed to load. Please check requirements.txt."
            st.error(error_msg) # Show error in UI
            return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
        if not openai or not hasattr(openai, 'OpenAI'): # Double check module/structure
             error_msg = "Internal Error: OpenAI library loaded flag is true, but module is None or has unexpected structure."
             print(error_msg)
             return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}


    # --- Actual API Call Logic ---
    try:
        # Ensure content is a string before formatting
        content_str = str(content) if content is not None else ""
        full_prompt = prompt_template.format(content=content_str)
        result_text = None

        if provider == 'gemini':
            # Configure API key for Gemini
            genai.configure(api_key=api_key)
            # Select Gemini model (make this configurable later if needed)
            model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using Flash for cost-effectiveness
            print(f"Calling Gemini model: {model.model_name}")
            # Define safety settings
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            # Specify JSON output
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
                safety_settings=safety_settings
                )
            # Handle potential safety blocks or empty responses
            # Check finish_reason as well
            if response.prompt_feedback.block_reason:
                 raise ValueError(f"Gemini response blocked. Reason: {response.prompt_feedback.block_reason}")
            if not response.parts:
                 raise ValueError("Gemini response was empty.")
            result_text = response.text
            print("Gemini call successful.")

        elif provider == 'openai':
            # Initialize OpenAI client with the API key
            client = openai.OpenAI(api_key=api_key)
            # Select OpenAI model (make this configurable later if needed)
            model_id = "gpt-3.5-turbo" # Or "gpt-4o", "gpt-4-turbo" etc.
            print(f"Calling OpenAI model: {model_id}")
            response = client.chat.completions.create(
               model=model_id,
               messages=[
                   {"role": "system", "content": "You are an SEO/LLMO analysis assistant outputting only valid JSON."},
                   {"role": "user", "content": full_prompt}
               ],
               response_format={ "type": "json_object" } # Request JSON output
            )
            # Add check for finish reason
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                 raise ValueError(f"OpenAI response did not finish normally. Reason: {finish_reason}")
            result_text = response.choices[0].message.content
            print("OpenAI call successful.")

        # --- Parse and Validate the JSON response ---
        if result_text:
            print(f"Raw LLM Response ({model_name}): {result_text[:300]}...") # Log more context
            parsed_result = json.loads(result_text)
            # Basic validation
            if not isinstance(parsed_result.get("score"), int):
                 raise ValueError("LLM response JSON format error: 'score' key missing or not an integer.")
            if not isinstance(parsed_result.get("recommendations"), list):
                raise ValueError("LLM response JSON format error: 'recommendations' key missing or not a list.")
            # Further validation on recommendation structure if needed
            for i, rec in enumerate(parsed_result.get("recommendations", [])):
                 if not isinstance(rec, dict):
                      raise ValueError(f"LLM response JSON format error: Recommendation item at index {i} is not a dictionary.")
                 if not isinstance(rec.get("text"), str) or not isinstance(rec.get("priority"), str):
                      raise ValueError(f"LLM response JSON format error: Recommendation item at index {i} missing 'text' or 'priority' string keys.")
            print("LLM response parsed and validated successfully.")
            return parsed_result
        else:
            # This case should be less likely now with checks on response parts/finish reason
            raise ValueError("LLM did not return any result text despite successful call.")

    # --- Error Handling ---
    except json.JSONDecodeError as e:
         error_msg = f"Failed to parse {model_name} response (Invalid JSON)."
         print(f"Error: {error_msg} - {e}. Response: {result_text[:500]}...") # Log response snippet
         st.warning(f"Could not parse analysis results from {model_name}. The LLM response was not valid JSON.")
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    except ValueError as e: # Catch validation errors, blocking errors, etc.
         error_msg = f"LLM response error from {model_name}: {e}"
         print(f"Error: {error_msg}")
         st.warning(f"Analysis results from {model_name} had an issue: {e}")
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    # Add specific exceptions for API errors if known from SDKs
    # except openai.AuthenticationError as e: ...
    # except genai.APIError as e: ... # Check actual exception types
    except Exception as e:
        error_msg = f"{model_name} API call failed: {type(e).__name__} - {e}"
        print(f"Error: {error_msg}")
        st.error(f"Error during {model_name} analysis. Please check API key, network connection, and model access permissions. Details: {e}")
        return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}


# --- Helper Functions for Free Checks (Keep the full code for these helpers as provided previously) ---
def _run_basic_text_checks(content):
    """Performs basic text analysis (chunking stats, sentiment, density)."""
    basic_recommendations = []
    if not nltk or not nltk_data_downloaded:
        basic_recommendations.append({"text": "NLTK setup incomplete, skipping basic text checks.", "priority": "Critical"})
        return basic_recommendations
    if not analyzer:
         basic_recommendations.append({"text": "VADER setup incomplete, skipping sentiment analysis.", "priority": "Critical"})

    try:
        # Ensure content is string
        content_str = str(content) if content is not None else ""
        if not content_str:
             basic_recommendations.append({"text": "[Basic Check Info] Content is empty, cannot perform text checks.", "priority": "Info"})
             return basic_recommendations

        # 1. Chunking Stats
        paragraphs = [p.strip() for p in content_str.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        sentences = sent_tokenize(content_str) # Uses NLTK's punkt
        num_sentences = len(sentences)
        avg_sent_per_para = round(num_sentences / num_paragraphs, 1) if num_paragraphs > 0 else 0
        long_sentences = [s for s in sentences if len(word_tokenize(s)) > 35] # Example threshold: 35 words
        basic_recommendations.append({
            "text": f"[Basic Check - Structure] Content has ~{num_paragraphs} paragraphs, {num_sentences} sentences (avg. {avg_sent_per_para} sentences/para). Found {len(long_sentences)} potentially long sentences (>35 words).",
            "priority": "Info"
        })

        # 2. Sentiment Analysis (VADER) - only if analyzer is ready
        if analyzer:
            vs = analyzer.polarity_scores(content_str)
            sentiment_label = "Neutral"
            if vs['compound'] >= 0.05: sentiment_label = "Positive"
            elif vs['compound'] <= -0.05: sentiment_label = "Negative"
            basic_recommendations.append({
                "text": f"[Basic Check - Sentiment] Overall Sentiment (VADER): {sentiment_label} (Compound Score: {vs['compound']:.2f})",
                "priority": "Info"
            })

        # 3. Keyword Density (Top 5 non-stopwords)
        stop_words = set(stopwords.words('english')) # Use NLTK's stopwords
        words = [word.lower() for word in word_tokenize(content_str) if word.isalpha() and len(word) > 1 and word.lower() not in stop_words] # Added len > 1
        total_words = len(words)
        if total_words > 20: # Require a minimum number of words for meaningful density
            word_counts = Counter(words)
            top_5_keywords = word_counts.most_common(5)
            density_report = ", ".join([f"'{k}' ({v} times, {((v/total_words)*100):.1f}%)" for k, v in top_5_keywords])
            basic_recommendations.append({
                "text": f"[Basic Check - Keywords] Top keywords (density): {density_report}",
                "priority": "Info"
            })
        else:
             basic_recommendations.append({"text": "[Basic Check - Keywords] Keyword density not calculated (insufficient non-stopword content).", "priority": "Info"})

    except Exception as e:
        print(f"Error during basic text checks: {e}")
        basic_recommendations.append({"text": f"[Basic Check Error] Could not perform basic text checks: {e}", "priority": "Critical"})

    return basic_recommendations

def _run_basic_html_checks(analysis_config):
    """Performs basic HTML structure checks using the soup object."""
    basic_recommendations = []
    soup = analysis_config.get("soup")
    url = analysis_config.get("input_url") # Get the original URL

    # Check prerequisites
    if not BeautifulSoup: # Check if library import failed
         basic_recommendations.append({"text": "BeautifulSoup library not found, skipping HTML checks.", "priority": "Critical"})
         return basic_recommendations
    if not soup or not isinstance(soup, BeautifulSoup):
        # This is expected for Raw Text input, so don't add an error, just return empty
        # basic_recommendations.append({"text": "[Basic Check Info] No HTML provided, skipping HTML checks.", "priority": "Info"})
        return basic_recommendations
    # Removed URL check here, urljoin handles None base better

    try:
        base_url = ""
        base_netloc = ""
        if url:
            try:
                base_url_parts = urlparse(url)
                # Handle potential //example.com URLs by adding scheme if missing
                scheme = base_url_parts.scheme or 'http' # Default to http if missing
                base_url = f"{scheme}://{base_url_parts.netloc}"
                base_netloc = base_url_parts.netloc
            except Exception as url_parse_error:
                 print(f"Could not parse base URL {url}: {url_parse_error}")
                 basic_recommendations.append({"text": f"[Basic Check Warning] Could not parse base URL '{url}', relative link analysis may be inaccurate.", "priority": "Medium"})


        # 1. HTML Structure Checks
        title_tag = soup.find('title')
        h1_tags = soup.find_all('h1')
        meta_desc_tag = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})

        title_text = title_tag.string.strip() if title_tag and title_tag.string else None
        if title_text:
            basic_recommendations.append({"text": f"[Basic Check - HTML] Title Tag found: '{title_text}' (Length: {len(title_text)})", "priority": "Info"})
        else:
            basic_recommendations.append({"text": "[Basic Check - HTML] Title Tag: Missing or empty.", "priority": "Medium"})

        if len(h1_tags) == 1:
             h1_text = h1_tags[0].get_text(strip=True)
             basic_recommendations.append({"text": f"[Basic Check - HTML] Single H1 Tag found: '{h1_text}'", "priority": "Info"})
        elif len(h1_tags) == 0:
             basic_recommendations.append({"text": "[Basic Check - HTML] H1 Tag: Missing.", "priority": "High"})
        else:
             basic_recommendations.append({"text": f"[Basic Check - HTML] H1 Tag: Found {len(h1_tags)} H1 tags (Ideally should be 1).", "priority": "Medium"})

        meta_desc_content = meta_desc_tag.get('content','').strip() if meta_desc_tag else None
        if meta_desc_content:
            basic_recommendations.append({"text": f"[Basic Check - HTML] Meta Description found (Length: {len(meta_desc_content)}).", "priority": "Info"})
        else:
            basic_recommendations.append({"text": "[Basic Check - HTML] Meta Description: Missing or empty.", "priority": "Medium"})

        # 2. Link Analysis
        internal_links = 0
        external_links = 0
        other_links = 0
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link['href']
            if not href or href.startswith('#') or href.startswith('javascript:'): # Skip empty, fragment, javascript links
                 other_links += 1
                 continue
            try:
                # Use the original URL as base if base_url parsing failed or wasn't provided
                current_base = base_url if base_netloc else url
                full_url = urljoin(current_base, href)
                parsed_href = urlparse(full_url)

                if parsed_href.scheme in ['http', 'https']:
                    # Compare netloc if base_netloc is available
                    if base_netloc and parsed_href.netloc == base_netloc:
                        internal_links += 1
                    elif base_netloc and parsed_href.netloc != base_netloc:
                        external_links += 1
                    else: # Fallback if base_netloc couldn't be parsed (treat as other)
                         other_links += 1
                elif parsed_href.scheme in ['mailto', 'tel']:
                     other_links +=1
                else: # Other schemes or potentially relative paths if base url failed badly
                    other_links += 1
            except Exception as link_parse_error:
                 print(f"Could not parse link {href}: {link_parse_error}")
                 other_links += 1 # Count as other if parsing fails
        basic_recommendations.append({"text": f"[Basic Check - Links] Links Found: {internal_links} Internal, {external_links} External, {other_links} Other/Fragment/Error.", "priority": "Info"})

        # 3. Image Alt Text Check
        img_tags = soup.find_all('img')
        imgs_missing_alt = 0
        imgs_empty_alt = 0
        for img in img_tags:
            alt_text = img.get('alt') # Get alt attribute
            if alt_text is None: # Alt attribute completely missing
                imgs_missing_alt += 1
            elif not alt_text.strip(): # Alt attribute exists but is empty or whitespace
                 imgs_empty_alt += 1

        total_imgs = len(img_tags)
        if total_imgs > 0:
             perc_missing = round(((imgs_missing_alt + imgs_empty_alt) / total_imgs) * 100)
             basic_recommendations.append({"text": f"[Basic Check - Images] Images Found: {total_imgs}. Missing/Empty Alt Text: {imgs_missing_alt + imgs_empty_alt} ({perc_missing}%).", "priority": "Info"})
             if imgs_missing_alt + imgs_empty_alt > 0:
                  basic_recommendations.append({"text": "[Basic Check - Images] Recommendation: Ensure all functional images have descriptive alt text.", "priority": "Medium"})
        else:
             basic_recommendations.append({"text": "[Basic Check - Images] No images (`<img>` tags) found on page.", "priority": "Info"})

    except Exception as e:
        print(f"Error during basic HTML checks: {e}")
        basic_recommendations.append({"text": f"[Basic Check Error] Could not perform basic HTML checks: {e}", "priority": "Critical"})

    return basic_recommendations


# --- Main Analysis Functions ---
# These combine basic checks with the LLM call

def analyze_content_chunking(content, analysis_config):
    """Analyzes Content Chunking using LLM + Basic Text Stats."""
    print(f"Analyzing Content Chunking...")
    basic_recs = _run_basic_text_checks(content)
    chunking_stats = [r for r in basic_recs if "[basic check - structure]" in r["text"].lower()]
    from .prompt_templates import CONTENT_CHUNKING_PROMPT
    llm_result = _call_llm_api(CONTENT_CHUNKING_PROMPT, content, analysis_config)
    # Prepend basic checks to LLM recommendations
    llm_result["recommendations"] = chunking_stats + llm_result.get("recommendations", [])
    return llm_result

def analyze_entity_presence(content, analysis_config):
    """Analyzes Entity Presence using LLM + Basic Density."""
    print(f"Analyzing Entity Presence...")
    basic_recs = _run_basic_text_checks(content)
    density_rec = [r for r in basic_recs if "[basic check - keywords]" in r["text"].lower()]
    from .prompt_templates import ENTITY_PRESENCE_PROMPT
    llm_result = _call_llm_api(ENTITY_PRESENCE_PROMPT, content, analysis_config)
    llm_result["recommendations"] = density_rec + llm_result.get("recommendations", [])
    return llm_result

def analyze_semantic_intent(content, analysis_config):
    """Analyzes Semantic Intent using LLM + Basic Sentiment."""
    print(f"Analyzing Semantic Intent...")
    basic_recs = _run_basic_text_checks(content)
    sentiment_rec = [r for r in basic_recs if "[basic check - sentiment]" in r["text"].lower()]
    from .prompt_templates import SEMANTIC_INTENT_PROMPT
    llm_result = _call_llm_api(SEMANTIC_INTENT_PROMPT, content, analysis_config)
    llm_result["recommendations"] = sentiment_rec + llm_result.get("recommendations", [])
    return llm_result

def analyze_structured_data(content, analysis_config):
    """Analyzes Structured Data using LLM + Basic HTML Checks."""
    print(f"Analyzing Structured Data...")
    basic_recs = _run_basic_html_checks(analysis_config)
    html_structure_recs = [r for r in basic_recs if "[basic check - html]" in r["text"].lower()]
    from .prompt_templates import STRUCTURED_DATA_PROMPT
    llm_result = _call_llm_api(STRUCTURED_DATA_PROMPT, content, analysis_config)
    llm_result["recommendations"] = html_structure_recs + llm_result.get("recommendations", [])
    return llm_result

def analyze_llm_parsing(content, analysis_config):
    """Analyzes LLM Parsing using LLM."""
    print(f"Analyzing LLM Parsing...")
    # Basic checks like sentence length included in 'analyze_content_chunking' results
    from .prompt_templates import LLM_PARSING_PROMPT
    llm_result = _call_llm_api(LLM_PARSING_PROMPT, content, analysis_config)
    return llm_result

def analyze_zero_click_signals(content, analysis_config):
    """Analyzes Zero Click Signals using LLM + Basic Link/Alt Text Checks."""
    print(f"Analyzing Zero Click Signals...")
    basic_recs = _run_basic_html_checks(analysis_config)
    link_alt_recs = [r for r in basic_recs if "[basic check - links]" in r["text"].lower() or "[basic check - images]" in r["text"].lower()]
    from .prompt_templates import ZERO_CLICK_SIGNALS_PROMPT
    llm_result = _call_llm_api(ZERO_CLICK_SIGNALS_PROMPT, content, analysis_config)
    llm_result["recommendations"] = link_alt_recs + llm_result.get("recommendations", [])
    return llm_result
