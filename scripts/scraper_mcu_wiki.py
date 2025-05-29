#!/usr/bin/env python3
import os
import time
import json
import requests
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# Configuration
API_URL          = "https://marvelcinematicuniverse.fandom.com/api.php"
RAW_DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'raw_mcu_wiki')
MOVIES_CATEGORY  = "Released_Movies"
REQUEST_DELAY    = 0.1  # seconds between requests
CHARACTERS       = [
    "Iron Man", "Captain America", "Thor", "Hulk", "Ant-Man", "Spider-Man", "Star-Lord", "Doctor Strange", "Black Panther", "Daredevil", "Black Widow", "Hawkeye", "Vision", "Scarlet Witch", "Falcon", "Winter Soldier", "War Machine", "Captain Marvel", "Thanos"
]


def ensure_data_dir():
    """
    Create the raw_mcu_wiki directory under data/raw.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    return RAW_DATA_DIR


def sanitize_filename(title: str) -> str:
    """Sanitize a title to be filesystem-safe."""
    return quote_plus(title.replace('/', '_'))


def fetch_category_members(session: requests.Session, category: str):
    """
    Yield (pageid, title) for all pages in a given fandom category.
    """
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'categorymembers',
        'cmtitle': f'Category:{category}',
        'cmlimit': '500',
        'cmnamespace': 0
    }
    while True:
        resp = session.get(API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        members = data.get('query', {}).get('categorymembers', [])
        for m in members:
            yield m['pageid'], m['title']
        if 'continue' in data:
            params.update(data['continue'])
            time.sleep(REQUEST_DELAY)
        else:
            break


def fetch_page_text(session: requests.Session, title: str) -> dict:
    """
    Fetch pageid, title, and cleaned plain-text content for a page via the parse API,
    collapsing all newlines and extra spaces into single spaces.
    """
    params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text',
        'redirects': 1
    }
    resp = session.get(API_URL, params=params)
    resp.raise_for_status()

    parse = resp.json().get('parse', {})
    pageid = int(parse.get('pageid', 0))
    title  = parse.get('title', title)
    html   = parse.get('text', {}).get('*', '')

    # 1) Strip HTML tags, getting all text in one big string
    raw = BeautifulSoup(html, 'html.parser').get_text()

    # 2) Remove non-ASCII
    raw = re.sub(r'[^\x00-\x7F]', '', raw)

    # 3) Collapse any run of whitespace (spaces, tabs, newlines) into a single space
    text = re.sub(r'\s+', ' ', raw).strip()

    return {
        'pageid': pageid,
        'title': title,
        'text': text
    }


def save_entry(entry: dict, directory: str):
    """
    Save an entry as a JSON file in the given directory.
    """
    filename = f"{entry['pageid']}_{sanitize_filename(entry['title'])}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


def main():
    raw_dir = ensure_data_dir()
    session = requests.Session()

    # --- Scrape Released Movies ---
    movie_list = list(fetch_category_members(session, MOVIES_CATEGORY))
    print(f"[MAIN] Found {len(movie_list)} released movies in Category:{MOVIES_CATEGORY}")
    for idx, (pageid, title) in enumerate(movie_list, start=1):
        entry = fetch_page_text(session, title)
        save_entry(entry, raw_dir)
        print(f"[SAVE] Movie {idx}/{len(movie_list)}: '{entry['title']}' ({entry['pageid']}) - text length: {len(entry['text'])}")
        time.sleep(REQUEST_DELAY)

    # --- Scrape Main Characters ---
    print(f"[MAIN] Scraping {len(CHARACTERS)} main characters")
    for idx, title in enumerate(CHARACTERS, start=1):
        entry = fetch_page_text(session, title)
        save_entry(entry, raw_dir)
        print(f"[SAVE] Character {idx}/{len(CHARACTERS)}: '{entry['title']}' ({entry['pageid']}) - text length: {len(entry['text'])}")
        time.sleep(REQUEST_DELAY)

    print("[MAIN] Completed scraping released movies and characters to raw_mcu_wiki.")


if __name__ == '__main__':
    main()
