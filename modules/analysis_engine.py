import time
import random
import streamlit as st
import re
import json # For parsing LLM responses
from collections import Counter
from urllib.parse import urlparse, urljoin

# --- Import libraries for Free Checks ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    st.error("NLTK library not found. Please install it (`pip install nltk`) and download necessary data.")
    nltk = None

try:
    from bs4 import BeautifulSoup
except ImportError:
     st.error("BeautifulSoup4 library not found. Please install it (`pip install beautifulsoup4`).")
     BeautifulSoup = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    st.error("vaderSentiment library not found. Please install it (`pip install vaderSentiment`).")
    SentimentIntensityAnalyzer = None

# --- Import LLM Libraries ---
# Add try-except blocks for robustness if deploying where installation might vary
try:
    import google.generativeai as genai
except ImportError:
    # Don't necessarily stop the app, maybe just disable Gemini option later
    st.warning("google-generativeai library not found. Gemini provider will not work.")
    genai = None

try:
    import openai
    # Check for version >= 1.0
    if hasattr(openai, 'OpenAI'):
        print("OpenAI library version >= 1.0 detected.")
    else:
        st.warning("Older OpenAI library version detected or library structure unexpected. Please use version >= 1.0.")
        # Handle compatibility or raise error if needed
except ImportError:
    st.warning("openai library not found. OpenAI provider will not work.")
    openai = None


# --- Download NLTK data (if needed) ---
# (Keeping the NLTK download logic as before)
nltk_data_downloaded = True
if nltk:
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError as e:
        missing_data = str(e).split("'")[1]
        st.info(f"Downloading NLTK '{missing_data}' data...")
        try:
            nltk.download(missing_data, quiet=True)
        except Exception as download_error:
             st.error(f"Failed to download NLTK data '{missing_data}': {download_error}")
             nltk_data_downloaded = False
    except Exception as general_nltk_error:
         st.error(f"An error occurred checking NLTK data: {general_nltk_error}")
         nltk_data_downloaded = False

# --- Initialize VADER ---
analyzer = None
if SentimentIntensityAnalyzer:
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        st.warning("VADER lexicon might be missing. Attempting NLTK download...")
        try:
            if nltk: nltk.download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer()
        except Exception as vader_download_error:
             st.error(f"Failed to download VADER lexicon via NLTK: {vader_download_error}")
    except Exception as vader_init_error:
        st.error(f"Failed to initialize VADER Sentiment Analyzer: {vader_init_error}")


# --- Central LLM API Call Function ---

def _call_llm_api(prompt_template, content, analysis_config):
    """
    Calls the selected LLM API (Gemini or OpenAI) and returns the parsed result.
    """
    provider = analysis_config.get("provider")
    api_key = analysis_config.get("api_key")
    model_name = provider.capitalize() if provider else "Unknown"

    print(f"--- Attempting LLM Call ({model_name}) ---")

    if not api_key or not provider:
         error_msg = f"Configuration Error: Missing API key or provider selection."
         print(error_msg)
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}

    # --- Actual API Call Logic ---
    try:
        full_prompt = prompt_template.format(content=content)
        result_text = None

        if provider == 'gemini':
            if not genai:
                 raise ImportError("Gemini library (google-generativeai) is not available.")
            # Configure API key for Gemini
            # Note: Configuring globally might have side effects if keys change often.
            # Consider managing client instances if needed.
            genai.configure(api_key=api_key)
            # Select Gemini model (make this configurable later if needed)
            model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using Flash for cost-effectiveness
            print(f"Calling Gemini model: {model.model_name}")
            # Set safety settings to be less restrictive if needed, handle potential blocks
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            # Specify JSON output if model supports it (Gemini 1.5 does)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
                safety_settings=safety_settings
                )
            # Handle potential safety blocks or empty responses
            if not response.parts:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 raise ValueError(f"Gemini response blocked or empty. Reason: {block_reason}")
            result_text = response.text
            print("Gemini call successful.")

        elif provider == 'openai':
            if not openai or not hasattr(openai, 'OpenAI'):
                 raise ImportError("OpenAI library (>= 1.0) is not available.")
            # Initialize OpenAI client with the API key
            client = openai.OpenAI(api_key=api_key)
            # Select OpenAI model (make this configurable later if needed)
            model_id = "gpt-3.5-turbo" # Or "gpt-4o", "gpt-4-turbo" etc.
            print(f"Calling OpenAI model: {model_id}")
            response = client.chat.completions.create(
               model=model_id,
               messages=[
                   # Providing system prompt might improve adherence to JSON format
                   {"role": "system", "content": "You are an SEO/LLMO analysis assistant outputting only valid JSON."},
                   {"role": "user", "content": full_prompt}
               ],
               response_format={ "type": "json_object" } # Request JSON output
            )
            result_text = response.choices[0].message.content
            print("OpenAI call successful.")
        else:
             raise ValueError(f"Unsupported LLM provider: {provider}")

        # --- Parse and Validate the JSON response ---
        if result_text:
            print(f"Raw LLM Response ({model_name}): {result_text[:200]}...") # Log snippet
            parsed_result = json.loads(result_text)
            # Basic validation
            if not isinstance(parsed_result.get("score"), int) or not isinstance(parsed_result.get("recommendations"), list):
                raise ValueError("LLM response JSON format error (score: int, recommendations: list).")
            for rec in parsed_result.get("recommendations", []):
                 if not isinstance(rec.get("text"), str) or not isinstance(rec.get("priority"), str):
                      raise ValueError("LLM response JSON format error (recommendation item: text/priority strings).")
            print("LLM response parsed successfully.")
            return parsed_result
        else:
            raise ValueError("LLM did not return any result text.")

    # --- Error Handling ---
    except json.JSONDecodeError as e:
         error_msg = f"Failed to parse {model_name} response (Invalid JSON)."
         print(f"Error: {error_msg} - {e}. Response: {result_text[:500]}...") # Log more context
         st.warning(f"Could not parse analysis results from {model_name}. The LLM response was not valid JSON.")
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    except ValueError as e: # Catch validation errors or specific issues like blocking
         error_msg = f"LLM response error from {model_name}: {e}"
         print(f"Error: {error_msg}")
         st.warning(f"Analysis results from {model_name} had an issue: {e}")
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    except ImportError as e:
         error_msg = f"Required library for {model_name} not found: {e}"
         print(f"Error: {error_msg}")
         st.error(error_msg) # Show error prominently in UI
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    except Exception as e:
        # Catch specific API errors from SDKs if possible (e.g., AuthenticationError, RateLimitError)
        # Example: Check for specific error types from google.api_core.exceptions or openai.error
        error_msg = f"{model_name} API call failed: {type(e).__name__} - {e}"
        print(f"Error: {error_msg}")
        st.error(f"Error during {model_name} analysis. Please check API key and network connection. Details: {e}")
        return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}


# --- Helper Functions for Free Checks (Unchanged from previous version) ---

def _run_basic_text_checks(content):
    """Performs basic text analysis (chunking stats, sentiment, density)."""
    basic_recommendations = []
    if not nltk or not nltk_data_downloaded:
        basic_recommendations.append({"text": "NLTK setup incomplete, skipping basic text checks.", "priority": "Critical"})
        return basic_recommendations
    if not analyzer:
         basic_recommendations.append({"text": "VADER setup incomplete, skipping sentiment analysis.", "priority": "Critical"})

    try:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        sentences = sent_tokenize(content)
        num_sentences = len(sentences)
        avg_sent_per_para = round(num_sentences / num_paragraphs, 1) if num_paragraphs > 0 else 0
        long_sentences = [s for s in sentences if len(word_tokenize(s)) > 35]
        basic_recommendations.append({
            "text": f"[Basic Check - Structure] Content has ~{num_paragraphs} paragraphs, {num_sentences} sentences (avg. {avg_sent_per_para} sentences/para). Found {len(long_sentences)} potentially long sentences (>35 words).",
            "priority": "Info"
        })

        if analyzer: # Only run if analyzer initialized
            vs = analyzer.polarity_scores(content)
            sentiment_label = "Neutral"
            if vs['compound'] >= 0.05: sentiment_label = "Positive"
            elif vs['compound'] <= -0.05: sentiment_label = "Negative"
            basic_recommendations.append({
                "text": f"[Basic Check - Sentiment] Overall Sentiment (VADER): {sentiment_label} (Compound Score: {vs['compound']:.2f})",
                "priority": "Info"
            })

        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in word_tokenize(content) if word.isalpha() and word.lower() not in stop_words]
        total_words = len(words)
        if total_words > 20:
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
    url = analysis_config.get("input_url")
    if not BeautifulSoup:
         basic_recommendations.append({"text": "BeautifulSoup library not found, skipping HTML checks.", "priority": "Critical"})
         return basic_recommendations
    if not soup or not isinstance(soup, BeautifulSoup): return basic_recommendations
    if not url:
        basic_recommendations.append({"text": "[Basic Check Info] Base URL not available, cannot fully analyze relative links.", "priority": "Info"})

    try:
        base_url = ""
        base_netloc = ""
        if url:
            try:
                base_url_parts = urlparse(url)
                base_url = f"{base_url_parts.scheme}://{base_url_parts.netloc}"
                base_netloc = base_url_parts.netloc
            except Exception as url_parse_error:
                 print(f"Could not parse base URL {url}: {url_parse_error}")
                 basic_recommendations.append({"text": f"[Basic Check Warning] Could not parse base URL '{url}', relative link analysis may be inaccurate.", "priority": "Medium"})

        # HTML Structure Checks (Title, H1, Desc)
        title_tag = soup.find('title')
        h1_tags = soup.find_all('h1')
        meta_desc_tag = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
        title_text = title_tag.string.strip() if title_tag and title_tag.string else None
        if title_text: basic_recommendations.append({"text": f"[Basic Check - HTML] Title Tag found: '{title_text}' (Length: {len(title_text)})", "priority": "Info"})
        else: basic_recommendations.append({"text": "[Basic Check - HTML] Title Tag: Missing or empty.", "priority": "Medium"})
        if len(h1_tags) == 1: basic_recommendations.append({"text": f"[Basic Check - HTML] Single H1 Tag found: '{h1_tags[0].get_text(strip=True)}'", "priority": "Info"})
        elif len(h1_tags) == 0: basic_recommendations.append({"text": "[Basic Check - HTML] H1 Tag: Missing.", "priority": "High"})
        else: basic_recommendations.append({"text": f"[Basic Check - HTML] H1 Tag: Found {len(h1_tags)} H1 tags (Ideally should be 1).", "priority": "Medium"})
        meta_desc_content = meta_desc_tag.get('content','').strip() if meta_desc_tag else None
        if meta_desc_content: basic_recommendations.append({"text": f"[Basic Check - HTML] Meta Description found (Length: {len(meta_desc_content)}).", "priority": "Info"})
        else: basic_recommendations.append({"text": "[Basic Check - HTML] Meta Description: Missing or empty.", "priority": "Medium"})

        # Link Analysis
        internal_links, external_links, other_links = 0, 0, 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href or href.startswith('#'): other_links += 1; continue
            try:
                full_url = urljoin(base_url or url, href)
                parsed_href = urlparse(full_url)
                if parsed_href.scheme in ['http', 'https']:
                    if base_netloc and parsed_href.netloc == base_netloc: internal_links += 1
                    elif base_netloc and parsed_href.netloc != base_netloc: external_links += 1
                    else: other_links += 1
                else: other_links += 1
            except Exception: other_links += 1
        basic_recommendations.append({"text": f"[Basic Check - Links] Links Found: {internal_links} Internal, {external_links} External, {other_links} Other/Fragment/Error.", "priority": "Info"})

        # Image Alt Text Check
        img_tags = soup.find_all('img')
        imgs_missing_alt, imgs_empty_alt = 0, 0
        for img in img_tags:
            alt_text = img.get('alt')
            if alt_text is None: imgs_missing_alt += 1
            elif not alt_text.strip(): imgs_empty_alt += 1
        total_imgs = len(img_tags)
        if total_imgs > 0:
             perc_missing = round(((imgs_missing_alt + imgs_empty_alt) / total_imgs) * 100)
             basic_recommendations.append({"text": f"[Basic Check - Images] Images Found: {total_imgs}. Missing/Empty Alt Text: {imgs_missing_alt + imgs_empty_alt} ({perc_missing}%).", "priority": "Info"})
             if imgs_missing_alt + imgs_empty_alt > 0: basic_recommendations.append({"text": "[Basic Check - Images] Recommendation: Ensure all functional images have descriptive alt text.", "priority": "Medium"})
        else: basic_recommendations.append({"text": "[Basic Check - Images] No images (`<img>` tags) found on page.", "priority": "Info"})

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
