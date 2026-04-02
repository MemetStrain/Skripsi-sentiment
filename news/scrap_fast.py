import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib3
from datetime import datetime
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

# --- CONFIGURATION ---
SCRAPE_LIMIT = 240000         # Stop after saving this many articles
MAX_WORKERS  = 8              # Reduced from 16 for stability (8-10 is safer)
FILENAME = "mpob_news_fast.csv"
SAVE_INTERVAL = 50            # Save to CSV every N articles (prevents data loss)
RETRY_ATTEMPTS = 3            # Retry failed requests up to 3 times
REQUEST_DELAY = 0.1           # Small delay between requests (seconds)

# DATE FILTERS
SCRAPE_START_DATE = None      # Will be auto-detected from existing file (or use "27-11-2025" if new)
SCRAPE_END_DATE   = "01-01-2001"  # Past (Cutoff)
# ---------------------

def get_latest_date_from_file(filename):
    """
    Load existing CSV and find the most recent date.
    Returns the latest date as a string in 'DD-MM-YYYY' format.
    """
    if not os.path.exists(filename):
        return None
    
    try:
        df = pd.read_csv(filename)
        if 'Date' not in df.columns or len(df) == 0:
            return None
        
        # Parse dates and find the most recent one
        df['DateParsed'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        latest_date = df['DateParsed'].max()
        
        if pd.isna(latest_date):
            return None
        
        # Return in DD-MM-YYYY format
        return latest_date.strftime('%d-%m-%Y')
    except Exception as e:
        print(f"[WARNING] Could not read existing file to detect latest date: {e}")
        return None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

base_urls = [
    "https://prestasisawit.mpob.gov.my/en/palmnews/advance-search/result?_token=YjvdaKzN9niwad2HbnwQZYgeEmK92dphtzmqeNuU&keyword=cpo",
    "https://prestasisawit.mpob.gov.my/en/palmnews/advance-search/result?_token=YjvdaKzN9niwad2HbnwQZYgeEmK92dphtzmqeNuU&keyword=crude%20palm%20oil"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

def create_session_with_retries():
    """Create a session with retry strategy and proper connection pooling."""
    session = requests.Session()
    session.headers.update(headers)
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=RETRY_ATTEMPTS,
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"]  # Only retry GET requests
    )
    
    # Configure HTTP adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=MAX_WORKERS * 2,  # Allow more connections
        pool_maxsize=MAX_WORKERS * 2,
        pool_block=False
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%d-%m-%Y")
    except ValueError:
        return None

def fetch_single_content(item):
    """Worker function: Creates its own session and fetches content with retries."""
    url = item['URL']
    
    # Each thread gets its own session (thread-safe)
    session = create_session_with_retries()
    
    try:
        # Small delay to avoid overwhelming server
        time.sleep(REQUEST_DELAY)
        
        # Increased timeout and specific timeout exceptions
        response = session.get(url, timeout=(10, 30), verify=False)
        # timeout=(connect, read) - 10s to connect, 30s to read
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.select_one('div.w-full.h-full.text-lg')
            if content_div:
                item['Content'] = content_div.get_text(separator='\n\n', strip=True)
            else:
                item['Content'] = "[Error: Content div not found]"
        else:
            item['Content'] = f"[Error: HTTP {response.status_code}]"
            
    except requests.exceptions.Timeout:
        item['Content'] = f"[Error: Timeout after {RETRY_ATTEMPTS} attempts]"
    except requests.exceptions.ConnectionError:
        item['Content'] = f"[Error: Connection failed]"
    except Exception as e:
        item['Content'] = f"[Error: {str(e)[:100]}]"  # Truncate long errors
    finally:
        session.close()  # Clean up session
    
    return item

def save_incremental(data, filename):
    """Save data incrementally to prevent data loss."""
    try:
        # Append to existing file or create new
        df = pd.DataFrame(data)
        if os.path.exists(filename):
            # Append mode
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            # New file
            df.to_csv(filename, index=False)
    except Exception as e:
        print(f"\n   [WARNING] Could not save incrementally: {e}")

def scrape_mpob_fast():
    all_news_data = []
    seen_urls = set()
    stop_scraping = False
    articles_since_last_save = 0

    # Auto-detect: scrape from TODAY until the latest date in existing file
    scrape_start_date = SCRAPE_START_DATE if SCRAPE_START_DATE else datetime.now().strftime('%d-%m-%Y')
    scrape_end_date = SCRAPE_END_DATE
    
    if os.path.exists(FILENAME):
        latest_date = get_latest_date_from_file(FILENAME)
        if latest_date:
            scrape_end_date = latest_date
            print(f"[AUTO-DETECT] Found existing file with latest date: {latest_date}")
            print(f"[MODE] Scraping from TODAY ({scrape_start_date}) backwards to {scrape_end_date}")
        else:
            print(f"[WARNING] Could not detect latest date from file, using default end date: {scrape_end_date}")
        
        # Load existing URLs to avoid duplicates
        try:
            existing_df = pd.read_csv(FILENAME)
            if 'URL' in existing_df.columns:
                seen_urls = set(existing_df['URL'].dropna().tolist())
                print(f"[LOADED] {len(seen_urls)} existing URLs to skip duplicates")
        except Exception as e:
            print(f"[WARNING] Could not load existing URLs: {e}")
    else:
        print(f"[NEW SCRAPE] No existing file found. Starting fresh from {scrape_start_date}")
    
    start_dt = parse_date(scrape_start_date)
    end_dt = parse_date(scrape_end_date)

    print(f"\n--- FAST SCRAPER (Optimized) ---")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Save Interval: Every {SAVE_INTERVAL} articles")
    print(f"Date Range: {scrape_start_date} (Today/Start) → {scrape_end_date} (Stop)")
    print("-------------------------------\n")

    # Session for listing pages (not shared with workers)
    session = create_session_with_retries()

    try:
        for start_url in base_urls:
            if stop_scraping: break
            
            current_url = start_url
            print(f"--- Processing Query: {start_url} ---")
            
            page_count = 1
            has_next_page = True

            while has_next_page:
                if stop_scraping: break

                try:
                    print(f"   [Page {page_count}] Scanning list...", end=" ")
                    response = session.get(current_url, timeout=(10, 30), verify=False)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    news_cards = soup.select('a > div.rounded.shadow-md')

                    if not news_cards:
                        print("No items found.")
                        break

                    # --- STEP 1: COLLECT VALID LINKS FIRST ---
                    batch_items = []
                    
                    for card_div in news_cards:
                        try:
                            # Date Check
                            date_tag = card_div.select_one('span.text-sm.text-black')
                            date_str = date_tag.get_text(strip=True) if date_tag else ""
                            current_date_obj = parse_date(date_str)

                            if current_date_obj:
                                if current_date_obj > start_dt: continue
                                if current_date_obj < end_dt:
                                    print(f"\n   [STOP] Reached cutoff date: {date_str}")
                                    stop_scraping = True
                                    break
                            
                            # Deduplicate with safe attribute access
                            link_tag = card_div.parent
                            if not link_tag or 'href' not in link_tag.attrs:
                                continue
                                
                            link_url = link_tag['href']
                            # Skip if URL already exists in the file
                            if link_url in seen_urls: 
                                continue
                            seen_urls.add(link_url)

                            # Safe extraction with fallbacks
                            title_tag = card_div.select_one('span.font-bold')
                            title = title_tag.get_text(strip=True) if title_tag else "No Title"
                            
                            snippet_tag = card_div.select_one('p.line-clamp-3')
                            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                            
                            cat_tag = card_div.select_one('span.divider-right')
                            category = cat_tag.get_text(strip=True) if cat_tag else "General"

                            batch_items.append({
                                'Date': date_str,
                                'Title': title,
                                'Category': category,
                                'Content': "Pending...", 
                                'Snippet': snippet,
                                'URL': link_url
                            })
                        except Exception as e:
                            print(f"\n   [WARNING] Error parsing card: {e}")
                            continue

                    if stop_scraping and not batch_items: break

                    print(f"Found {len(batch_items)} valid items. Downloading content...", end=" ")

                    # --- STEP 2: PARALLEL DOWNLOAD CONTENT (with per-thread sessions) ---
                    if batch_items:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            # Submit all tasks (no shared session)
                            futures = [executor.submit(fetch_single_content, item) for item in batch_items]
                            
                            # Wait for completion with timeout per future
                            for future in concurrent.futures.as_completed(futures, timeout=300):
                                try:
                                    future.result()  # This will raise any exceptions
                                except concurrent.futures.TimeoutError:
                                    print(f"\n   [WARNING] A request timed out")
                                except Exception as e:
                                    print(f"\n   [WARNING] Worker error: {e}")
                        
                        # Add processed batch to main list
                        all_news_data.extend(batch_items)
                        articles_since_last_save += len(batch_items)
                        
                        print(f"Done. (Total: {len(all_news_data)})")
                        
                        # Incremental save
                        if articles_since_last_save >= SAVE_INTERVAL:
                            print(f"   [AUTO-SAVE] Saving {len(all_news_data)} articles...", end=" ")
                            save_incremental(batch_items, FILENAME)
                            articles_since_last_save = 0
                            print("Saved.")
                    else:
                        print("Skipped (Duplicates or outside date range).")

                    # --- STEP 3: CHECK LIMIT ---
                    if SCRAPE_LIMIT and len(all_news_data) >= SCRAPE_LIMIT:
                        print(f"\n   [STOP] Hit limit of {SCRAPE_LIMIT}.")
                        stop_scraping = True
                        break

                    # --- STEP 4: PAGINATION ---
                    if not stop_scraping:
                        next_btn = soup.find('a', attrs={'rel': 'next'})
                        if next_btn and next_btn.has_attr('href'):
                            current_url = next_btn['href']
                            page_count += 1
                            time.sleep(0.3)  # Small delay between pages
                        else:
                            has_next_page = False
                
                except requests.exceptions.Timeout:
                    print(f"\n   [RETRY] Timeout on Page {page_count}, retrying...")
                    time.sleep(2)
                    # Don't break, will retry on next iteration
                except requests.exceptions.RequestException as e:
                    print(f"\n   Error on Page {page_count}: {e}")
                    time.sleep(2)  # Wait before retrying
                except Exception as e:
                    print(f"\n   Unexpected error on Page {page_count}: {e}")
                    time.sleep(2)
    finally:
        session.close()

    # FINAL SAVE (all remaining data)
    if all_news_data:
        # Load existing if incremental saves happened
        if os.path.exists(FILENAME):
            try:
                existing_df = pd.read_csv(FILENAME)
                new_df = pd.DataFrame(all_news_data)
                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['URL'], keep='last')
                # Sort by date (newest first)
                combined_df['DateParsed'] = pd.to_datetime(combined_df['Date'], format='%d-%m-%Y', errors='coerce')
                combined_df = combined_df.sort_values('DateParsed', ascending=False).drop(columns=['DateParsed'])
                combined_df.to_csv(FILENAME, index=False)
                print(f"\n\nSuccess! Total articles in {FILENAME}: {len(combined_df)}")
                print(f"New articles added: {len(new_df)}")
            except Exception as e:
                print(f"\n[WARNING] Could not merge with existing file: {e}")
                # Fallback: save new data separately
                df = pd.DataFrame(all_news_data)
                df.to_csv(FILENAME.replace('.csv', '_new.csv'), index=False)
                print(f"Saved new data to {FILENAME.replace('.csv', '_new.csv')}")
        else:
            df = pd.DataFrame(all_news_data)
            df.to_csv(FILENAME, index=False)
            print(f"\n\nSuccess! Saved {len(df)} articles to {FILENAME}")
    else:
        print("\nNo new articles extracted (all may be duplicates or outside date range).")

if __name__ == "__main__":
    scrape_mpob_fast()