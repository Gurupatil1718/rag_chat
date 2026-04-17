"""
Website scraper for mitratechgroup.com.

Uses requests + BeautifulSoup to crawl key pages and return clean text.
All scraping happens at startup (or on-demand via /ingest endpoint).
"""

import logging
import re
import time
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Pages to crawl (relative paths on mitratechgroup.com)
CRAWL_PATHS: List[str] = [
    "/",
    "/about",
    "/services",
    "/resource-augmentation",
    "/contact",
    "/solutions",
    "/ai-solutions",
    "/web-development",
    "/cybersecurity",
    "/blockchain",
    "/enterprise",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MitraTechBot/1.0; "
        "+https://www.mitratechgroup.com)"
    )
}


def _clean_text(raw: str) -> str:
    """Normalise whitespace and strip boilerplate noise."""
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", raw)
    # Collapse multiple blank lines to a single one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _scrape_page(url: str, timeout: int = 10) -> str:
    """Fetch a single URL and return cleaned visible text."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg", "img",
                     "head", "footer", "nav", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return _clean_text(text)


def scrape_website(base_url: str) -> str:
    """
    Crawl predefined pages on *base_url* and return a single concatenated
    string ready for chunking.
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    collected: List[str] = []
    seen: set = set()

    for path in CRAWL_PATHS:
        url = urljoin(base, path)
        if url in seen:
            continue
        seen.add(url)

        logger.info("Scraping: %s", url)
        page_text = _scrape_page(url)
        if page_text:
            collected.append(f"## Source: {url}\n\n{page_text}")

        time.sleep(0.5)   # polite delay

    full_content = "\n\n---\n\n".join(collected)
    logger.info(
        "Scraping complete. Total characters collected: %d", len(full_content)
    )
    return full_content


def load_local_data(file_path: str) -> str:
    """Read pre-saved content from a local text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return _clean_text(fh.read())
    except FileNotFoundError:
        logger.warning("Local data file not found: %s", file_path)
        return ""
