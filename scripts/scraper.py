#!/usr/bin/env python3
import os
import time
import json
import requests
from urllib.parse import quote_plus

# Configuration
API_URL        = "https://marvel.fandom.com/api.php"
RAW_DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CATEGORY       = "Films"
CM_LIMIT       = 500     # max pages per request
REQUEST_DELAY  = 0.1     # seconds between requests


def ensure_data_dir():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def sanitize_filename(title: str) -> str:
    return quote_plus(title.replace('/', '_'))


def fetch_category_members(session):
    """
    Iterate over all pages in Category:Films, yielding (pageid, title).
    """
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'categorymembers',
        'cmtitle': f'Category:{CATEGORY}',
        'cmlimit': CM_LIMIT,
        'cmnamespace': 0
    }
    total = 0
    batch = 1
    while True:
        resp = session.get(API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        members = data.get('query', {}).get('categorymembers', [])
        count = len(members)
        total += count
        print(f"[LIST] Batch {batch}: Retrieved {count} pages (total so far: {total})")
        for m in members:
            yield m['pageid'], m['title']
        if 'continue' in data:
            params.update(data['continue'])
            batch += 1
            time.sleep(REQUEST_DELAY)
        else:
            print(f"[LIST] Completed listing {total} pages in Category:{CATEGORY}")
            break


def fetch_wikitext(session, pageid):
    """
    Fetch the full wikitext content of a page by its pageid.
    """
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'revisions',
        'pageids': pageid,
        'rvprop': 'content',
        'rvslots': 'main'
    }
    resp = session.get(API_URL, params=params)
    resp.raise_for_status()
    page = resp.json().get('query', {}).get('pages', {}).get(str(pageid), {})
    revisions = page.get('revisions', [])
    text = revisions[0]['slots']['main']['*'] if revisions else ''
    print(f"[WIKI] pageid={pageid}: fetched {len(text)} characters")
    return text


def main():
    ensure_data_dir()
    session = requests.Session()

    # 1) List all films
    pages = list(fetch_category_members(session))
    total = len(pages)
    print(f"[MAIN] Total films found: {total}")

    # 2) Fetch and save each page's wikitext
    for idx, (pageid, title) in enumerate(pages, start=1):
        text = fetch_wikitext(session, pageid)
        filename = f"{pageid}_{sanitize_filename(title)}.json"
        filepath = os.path.join(RAW_DATA_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'pageid': pageid, 'title': title, 'text': text}, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {idx}/{total}: Saved '{title}' ({pageid})")
        time.sleep(REQUEST_DELAY)

    print(f"[MAIN] Completed saving {total} files to {RAW_DATA_DIR}")


if __name__ == '__main__':
    main()
