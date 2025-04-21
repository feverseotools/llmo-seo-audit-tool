import time
import random
import streamlit as st
import re
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

# --- Download NLTK data (if library is available and data not found) ---
nltk_data_downloaded = True
if nltk:
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        # Check for VADER lexicon via NLTK downloader as well, though VADER often handles its own download
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError as e:
        missing_data = str(e).split("'")[1] # Extract missing package name
        st.info(f"Downloading NLTK '{missing_data}' data...")
        try:
            nltk.download(missing_data, quiet=True)
        except Exception as download_error:
             st.error(f"Failed to download NLTK data '{missing_data}': {download_error}")
             nltk_data_downloaded = False # Flag that setup might be incomplete
    except Exception as general_nltk_error:
         st.error(f"An error occurred checking NLTK data: {general_nltk_error}")
         nltk_data_downloaded = False


# --- Initialize VADER (if library is available) ---
analyzer = None
if SentimentIntensityAnalyzer:
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        # This might happen if the NLTK download above didn't cover it or failed
        st.warning("VADER lexicon might be missing. Attempting NLTK download again...")
        try:
            nltk.download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer() # Try initializing again
        except Exception as vader_download_error:
             st.error(f"Failed to download VADER lexicon via NLTK: {vader_download_error}")
    except Exception as vader_init_error:
        st.error(f"Failed to initialize VADER Sentiment Analyzer: {vader_init_error}")


# --- LLM API Call Function (Placeholder - unchanged) ---
def _call_llm_api(prompt_template, content, analysis_config):
    """
    Placeholder function to simulate calling the selected LLM API.
    Replace this with actual API call logic using the provided key and provider.
    """
    provider = analysis_config.get("provider")
    api_key = analysis_config.get("api_key")
    model_name = provider.capitalize() if provider else "Unknown"
    print(f"--- Simulating LLM Call ({model_name}) ---")
    if not api_key or not provider:
         error_msg = f"Configuration Error: Missing API key or provider selection."
         print(error_msg)
         return {"score": 0, "recommendations": [{"text": error_msg, "priority": "Critical"}]}
    else:
         print(f"Using {model_name} Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    print(f"------------------------------------")
    time.sleep(random.uniform(0.5, 1.5)) # Shorter simulation delay

    # --- Dummy Data Generation ---
    print("--- Using Dummy Data for LLM ---")
    score = random.randint(40, 95)
    dummy_recommendations = [
        {"text": f"This is a placeholder LLM recommendation ({model_name}).", "priority": random.choice(["Low", "Medium", "High"])},
    ]
    return {"score": score, "recommendations": dummy_recommendations}
    # --- End Dummy Data ---


# --- Helper Functions for Free Checks ---

def _run_basic_text_checks(content):
    """Performs basic text analysis (chunking stats, sentiment, density)."""
    basic_recommendations = []

    # Check if NLTK and VADER are available and ready
    if not nltk or not nltk_data_downloaded:
        basic_recommendations.append({"text": "NLTK setup incomplete, skipping basic text checks.", "priority": "Critical"})
        return basic_recommendations
    if not analyzer:
         basic_recommendations.append({"text": "VADER setup incomplete, skipping sentiment analysis.", "priority": "Critical"})
         # Continue with other checks if possible

    try:
        # 1. Chunking Stats
        # Simple split by double newline, might need refinement for different content structures
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        sentences = sent_tokenize(content) # Uses NLTK's punkt
        num_sentences = len(sentences)
        avg_sent_per_para = round(num_sentences / num_paragraphs, 1) if num_paragraphs > 0 else 0
        # Add check for very long sentences
        long_sentences = [s for s in sentences if len(word_tokenize(s)) > 35] # Example threshold: 35 words
        basic_recommendations.append({
            "text": f"[Basic Check - Structure] Content has ~{num_paragraphs} paragraphs, {num_sentences} sentences (avg. {avg_sent_per_para} sentences/para). Found {len(long_sentences)} potentially long sentences (>35 words).",
            "priority": "Info"
        })

        # 2. Sentiment Analysis (VADER) - only if analyzer is ready
        if analyzer:
            vs = analyzer.polarity_scores(content)
            sentiment_label = "Neutral"
            if vs['compound'] >= 0.05: sentiment_label = "Positive"
            elif vs['compound'] <= -0.05: sentiment_label = "Negative"
            basic_recommendations.append({
                "text": f"[Basic Check - Sentiment] Overall Sentiment (VADER): {sentiment_label} (Compound Score: {vs['compound']:.2f})",
                "priority": "Info"
            })

        # 3. Keyword Density (Top 5 non-stopwords)
        stop_words = set(stopwords.words('english')) # Use NLTK's stopwords
        words = [word.lower() for word in word_tokenize(content) if word.isalpha() and word.lower() not in stop_words]
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
        return basic_recommendations # Cannot run these checks without soup object
    if not url: # Need base URL for joining relative links
        basic_recommendations.append({"text": "[Basic Check Info] Base URL not available, cannot fully analyze relative links.", "priority": "Info"})
        # Continue with other checks that don't rely on base URL

    try:
        # Parse base URL safely
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
            if not href or href.startswith('#'): # Skip empty or fragment links
                 other_links += 1
                 continue
            try:
                full_url = urljoin(base_url or url, href) # Use base_url if available, else original url
                parsed_href = urlparse(full_url)

                if parsed_href.scheme in ['http', 'https']:
                    # Check netloc against base_netloc if available
                    if base_netloc and parsed_href.netloc == base_netloc:
                        internal_links += 1
                    elif base_netloc and parsed_href.netloc != base_netloc:
                        external_links += 1
                    else: # Fallback if base_netloc couldn't be parsed
                         other_links += 1
                else:
                    other_links += 1 # mailto, tel, javascript, etc.
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


# --- Main Analysis Functions (Updated to include basic checks) ---

def analyze_content_chunking(content, analysis_config):
    """Analyzes Content Chunking using LLM + Basic Text Stats."""
    print(f"Analyzing Content Chunking...")
    # Run basic text checks first (includes chunking stats)
    basic_recs = _run_basic_text_checks(content) # Gets para/sent counts, sentiment, density
    chunking_stats = [r for r in basic_recs if "[basic check - structure]" in r["text"].lower()] # Extract only structure stats

    # Call LLM for deeper analysis
    from .prompt_templates import CONTENT_CHUNKING_PROMPT
    llm_result = _call_llm_api(CONTENT_CHUNKING_PROMPT, content, analysis_config)

    # Combine results
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
    # Run basic HTML checks (Title, H1, Desc) - requires soup object
    basic_recs = _run_basic_html_checks(analysis_config)
    html_structure_recs = [r for r in basic_recs if "[basic check - html]" in r["text"].lower()]

    # Call LLM (focuses on *opportunities* in text)
    from .prompt_templates import STRUCTURED_DATA_PROMPT
    llm_result = _call_llm_api(STRUCTURED_DATA_PROMPT, content, analysis_config)

    # Combine
    llm_result["recommendations"] = html_structure_recs + llm_result.get("recommendations", [])
    return llm_result

def analyze_llm_parsing(content, analysis_config):
    """Analyzes LLM Parsing using LLM (+ Readability as proxy if added)."""
    print(f"Analyzing LLM Parsing...")
    # Basic text checks (e.g., sentence length) are done in analyze_content_chunking now
    # Readability score could be added here using textstat if installed

    from .prompt_templates import LLM_PARSING_PROMPT
    llm_result = _call_llm_api(LLM_PARSING_PROMPT, content, analysis_config)
    # Add readability score if implemented:
    # readability_recs = _calculate_readability(content) # Needs implementation
    # llm_result["recommendations"] = readability_recs + llm_result.get("recommendations", [])
    return llm_result


def analyze_zero_click_signals(content, analysis_config):
    """Analyzes Zero Click Signals using LLM + Basic Link/Alt Text Checks."""
    print(f"Analyzing Zero Click Signals...")
    # Run basic HTML checks (includes Links, Alt Text) - requires soup
    basic_recs = _run_basic_html_checks(analysis_config)
    link_alt_recs = [r for r in basic_recs if "[basic check - links]" in r["text"].lower() or "[basic check - images]" in r["text"].lower()]

    # Call LLM (focuses on formatting like lists, headings in text)
    from .prompt_templates import ZERO_CLICK_SIGNALS_PROMPT
    llm_result = _call_llm_api(ZERO_CLICK_SIGNALS_PROMPT, content, analysis_config)

    # Combine
    llm_result["recommendations"] = link_alt_recs + llm_result.get("recommendations", [])
    return llm_result
