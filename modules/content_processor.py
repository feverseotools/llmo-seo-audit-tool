import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse # Needed for link analysis later if done here

# Consider adding more robust extraction libraries later, e.g.:
# from readability import Document

def fetch_content_from_url(url):
    """Fetches HTML, extracts main text content, and returns both text and the BeautifulSoup object.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        tuple: (extracted_text, soup_object) on success, or (None, None) on failure.
               soup_object will be None if fetching/parsing fails.
    """
    if not url or not (url.startswith('http://') or url.startswith('https://')):
        st.error("Invalid URL provided. Please include http:// or https://")
        return None, None

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            st.error(f"Expected HTML content, but received content type: {content_type}")
            return None, None

        # --- Parse the full HTML with BeautifulSoup ---
        soup = BeautifulSoup(response.text, 'html.parser')
        if not soup.body:
             st.error("Could not find body tag in the HTML.")
             return None, soup # Return soup even if body is missing, maybe head is useful?

        # --- Extract Main Text Content ---
        # Create a copy of the soup to avoid modifying the original when removing tags for text extraction
        text_soup = BeautifulSoup(response.text, 'html.parser')

        # Remove common non-content tags from the copy
        for tag in text_soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'link', 'meta', 'noscript', 'svg', 'img', 'figure']):
            tag.decompose()

        # Attempt to find main content containers in the copy
        main_content_area = text_soup.find('article') or text_soup.find('main') or text_soup.find(role='main') or text_soup.find('div', id='content') or text_soup.find('div', class_='content')

        text_extraction_source = main_content_area if main_content_area else text_soup.body

        if not text_extraction_source:
             # Should not happen if soup.body exists, but as a fallback
             st.warning("Could not identify main content area or body for text extraction.")
             cleaned_text = ""
        else:
             # Extract text from relevant tags within the identified area
             text_parts = [elem.get_text(separator=' ', strip=True) for elem in text_extraction_source.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'span', 'div'])]
             full_text = ' '.join(filter(None, text_parts))
             cleaned_text = ' '.join(full_text.split())

        # Basic check for meaningful content length
        if len(cleaned_text) < 50:
             st.warning(f"Extracted text is very short ({len(cleaned_text)} chars). Analysis quality may be affected.")
             # If main content extraction failed, try getting text from the whole body as fallback
             if not main_content_area and soup.body:
                  body_text = ' '.join(soup.body.get_text(separator=' ', strip=True).split())
                  if len(body_text) > len(cleaned_text):
                      st.info("Using text from entire body as fallback.")
                      cleaned_text = body_text


        # Return the cleaned text and the original, unmodified soup object
        return cleaned_text.strip(), soup

    except requests.exceptions.Timeout:
        st.error(f"Error fetching URL: Request timed out after 15 seconds.")
        return None, None
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching URL: HTTP Error {e.response.status_code} for url: {url}")
        return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: Network error - {e}")
        return None, None
    except Exception as e:
        st.error(f"Error parsing content: {e}")
        # Return None for text, but maybe the soup object is partially valid? Or None too? Let's return None.
        return None, None
