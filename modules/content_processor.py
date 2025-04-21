import streamlit as st # Import streamlit for st.error
import requests
from bs4 import BeautifulSoup
# Consider adding more robust extraction libraries later, e.g.:
# from readability import Document

def fetch_content_from_url(url):
    """Fetches and extracts main content from a URL.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: The extracted text content, or None if an error occurs.
    """
    if not url or not url.startswith(('http://', 'https://')):
        st.error("Invalid URL provided.")
        return None

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            # Using a realistic user agent
        }
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- More Robust Content Extraction (Example using BeautifulSoup) ---
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove common non-content tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe']):
            tag.decompose()

        # Attempt to find common main content containers
        main_content = soup.find('article') or soup.find('main') or soup.find(role='main')

        if not main_content:
             # Fallback to body if specific containers aren't found
             main_content = soup.body
             if not main_content:
                 st.error("Could not find body tag in the HTML.")
                 return None # No body tag found

        # Extract text from relevant tags within the main content area
        # Prioritize p, headings, li. Add others if needed.
        text_parts = [p.get_text(separator=' ', strip=True) for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'])]

        # Join the parts and clean up whitespace
        full_text = ' '.join(text_parts)
        cleaned_text = ' '.join(full_text.split()) # Remove extra whitespace

        if not cleaned_text:
             st.warning("Extracted content appears to be empty after cleaning.")
             # Try getting text directly from body as a last resort if main_content logic failed
             if soup.body:
                 cleaned_text = ' '.join(soup.body.get_text(separator=' ', strip=True).split())
                 if cleaned_text:
                     st.info("Used fallback body text extraction.")
                 else:
                     st.error("Could not extract meaningful text content.")
                     return None
             else:
                 st.error("Could not extract meaningful text content.")
                 return None


        return cleaned_text.strip()
        # --- End Robust Extraction ---

    except requests.exceptions.Timeout:
        st.error(f"Error fetching URL: Request timed out after 15 seconds.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching URL: HTTP Error {e.response.status_code} for url: {url}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        # Catch other potential errors during parsing
        st.error(f"Error parsing content: {e}")
        return None

# You could add more functions here for text cleaning, preprocessing, etc.
# def preprocess_text(text):
#     # Example: Lowercasing, removing special characters (use cautiously)
#     text = text.lower()
#     # ... more steps ...
#     return text
