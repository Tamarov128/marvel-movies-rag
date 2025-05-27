#!/usr/bin/env python3
import os
import time
import json
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# Configuration
API_URL       = "https://en.wikipedia.org/w/api.php"
RAW_DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'raw_wikipedia')
REQUEST_DELAY = 0.1  # seconds between requests
TARGET_PAGE   = "List_of_films_based_on_Marvel_Comics_publications"


def ensure_data_dir():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def sanitize_filename(title: str) -> str:
    return quote_plus(title.replace('/', '_'))


def fetch_movie_text(session: requests.Session, title: str) -> dict:
    """
    Uses the extracts API to fetch plain-text summary for a given film title.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title.replace(" ", "_"),
        "redirects": 1,
        "explaintext": 1,
        "exsectionformat": "plain"
    }
    resp = session.get(API_URL, params=params)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    pageid, page = next(iter(pages.items()))
    return {
        "pageid": int(pageid),
        "title": page.get("title"),
        "text": page.get("extract", "")
    }


def get_feature_film_titles(session: requests.Session) -> list:
    """
    Parses the 'Feature films' section of the target List page to extract all film titles.
    """
    # 1) Fetch sections to find the index of 'Feature films'
    sec_params = {
        "action": "parse",
        "page": TARGET_PAGE,
        "format": "json",
        "prop": "sections"
    }
    sec_resp = session.get(API_URL, params=sec_params)
    sec_resp.raise_for_status()
    sections = sec_resp.json().get("parse", {}).get("sections", [])
    feature_index = None
    for sec in sections:
        if sec.get("line", "").strip().lower() == "feature films":
            feature_index = sec.get("index")
            break
    if feature_index is None:
        raise RuntimeError("Could not find 'Feature films' section on the page.")

    # 2) Fetch HTML of that section
    html_params = {
        "action": "parse",
        "page": TARGET_PAGE,
        "format": "json",
        "prop": "text",
        "section": feature_index
    }
    html_resp = session.get(API_URL, params=html_params)
    html_resp.raise_for_status()
    html = html_resp.json().get("parse", {}).get("text", {}).get("*", "")

    # 3) Parse the HTML table for film links
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="wikitable")
    if not table:
        raise RuntimeError("No wikitable found in 'Feature films' section.")

    titles = []
    # first column has link in <th> or <td>
    for row in table.find_all("tr")[1:]:  # skip header row
        cell = row.find(["th", "td"])
        link = cell.find("a") if cell else None
        if link and link.has_attr("title"):
            titles.append(link["title"])
    return titles


def main():
    ensure_data_dir()
    session = requests.Session()

    # 1) Gather all feature-film titles
    titles = get_feature_film_titles(session)
    total = len(titles)
    print(f"[MAIN] Total films found: {total}")

    # 2) Fetch each film's extract and save to JSON
    for idx, title in enumerate(titles, start=1):
        movie = fetch_movie_text(session, title)
        filename = f"{movie['pageid']}_{sanitize_filename(movie['title'])}.json"
        filepath = os.path.join(RAW_DATA_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(movie, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {idx}/{total}: Saved '{movie['title']}' ({movie['pageid']})")
        time.sleep(REQUEST_DELAY)

    print(f"[MAIN] Completed saving {total} files to {RAW_DATA_DIR}")


if __name__ == '__main__':
    main()
