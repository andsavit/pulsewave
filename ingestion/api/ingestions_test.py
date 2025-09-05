import time, random, re 
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timezone
import logging
import os
import pyarrow
from dotenv import load_dotenv
#REQUEST SETTINGS
BASE = "https://reverb.com/"

load_dotenv()

def make_session():
    session = requests.Session()
    session.headers.update({
        "Accept": "application/hal+json",
        "Accept-Version": "3.0",
        "Accept-Language": "it-IT",
        "X-Display-Currency": "EUR",
        "X-Shipping-Region": "IT", #FILTERS RESULTS!
        "Authorization": f"Bearer {os.getenv("REVERB_TOKEN")}", #should get token from .env file REVERB_TOKEN
        "User-Agent": "reverb-application-updater/1.2"
    })
    session.params.update({
        "condition": "used",
        "per_page": 50
    })
    return session

HEADERS = {
        "Accept": "application/hal+json",
        "Accept-Version": "3.0",
        "Accept-Language": "it-IT",
        "X-Display-Currency": "EUR",
        "X-Shipping-Region": "IT", #FILTERS RESULTS!
        "Authorization": f"Bearer {os.getenv("REVERB_TOKEN")}", #should get token from .env file REVERB_TOKEN
        "User-Agent": "reverb-application-updater/1.2"
}

params = {
        "condition": "used",
        "per_page": 50
    }
#INIT====================================
snapshot_ts = datetime.now(timezone.utc)

#LOGGING
if params:
    condition = params.get('condition', '')
else: condition = ''

if HEADERS:
    shprg = HEADERS.get('X-Shipping-Region', '')
else: shprg = ''

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/reverb_scan_{condition}-{snapshot_ts.strftime('%Y%m%d-%H%M')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
) 

#level setting
root_logger = logging.getLogger()
root_logger.handlers[0].setLevel(logging.DEBUG)
root_logger.handlers[1].setLevel(logging.INFO)

logger = logging.getLogger(__name__)


####FILTERING AND FIELD SELECTION/CLEANING====================================
def filter_listing(x, product_type, category):
    """
    Input: a single listing
    Output: a listing with selected, flattened fields
    """
    return {
        #flat fields----------------------------------------------------------
        "id": x['id'],
        "make": x.get('make', ""),
        'model': x.get('model', ''),
        'finish': x.get('finish', ''),
        'year': x.get('year', ''),
        'sku': x.get('sku', ''),
        'product_type': product_type,
        'category': category,
        'title': x.get('title', ''),
        'created_at': to_utc(x.get('created_at', '')),
        'shop_slug': x.get('shop', {}).get('slug', ''),
        'preferred_seller': x.get('shop', {}).get('preferred_seller', ''),
        'condition': x.get('condition_slug', {}).get('slug', ''),
        'offers_enabled': x.get('offers_enabled', ''),
        'has_inventory': x.get('has_inventory'),
        'inventory': x.get('inventory', None),
        'published_at': to_utc(x.get('published_at', '')),
        'state': x.get('state', {}).get('slug', ''),
        'auction': x.get('auction', ''),
        'permalink': str(x.get('_links', {}).get('self', {}).get('href', '')),
        #flattened fields------------------------------------------------------
        'price': x.get('price', {}).get('amount', ''),
        'price_currency': x.get('price', {}).get('currency', ''),
        'price_taxIncluded': x.get('price', {}).get('tax_included', ''),
        'buyer_price': x.get('buyer_price', {}).get('amount', ''),
        'buyer_price_currency': x.get('buyer_price', {}).get('currency', ''),
        'buyer_price_taxIncluded': x.get('buyer_price', {}).get('tax_included', ''),
        #calculated fields---------------------------------------------
        #description
        #'description': clean_description(x.get('description', ''))
        #shipping: to define based on modeling perspective. Option if modeled in respect to IT: if IT is present, take that price, otherwise region "XX" price
        # SCD 2 Fields---------------------------------------------
        'snap_valid_from': None,   # snapshot_ts quando entra in questa versione
        'snap_valid_to': None,     # NULL finché corrente
        'snap_is_current': None
    }

#NORMALIZING FUNCTIONS
def to_utc(s: str):
    """
    Convert Reverb's ISO8601 with offset (e.g. '2023-04-19T14:03:11-05:00')
    into a UTC datetime object.
    """
    if not s:
        return None
    dt = datetime.fromisoformat(s)          # parse with offset
    return dt.astimezone(timezone.utc)      # normalize to UTC

def clean_description(s: str) -> str:
    # unescape HTML entities and remove the most common tags quickly
    if not s:
        return ""
    #s = unescape(s)
    # quick & dirty tag strip (good enough for Reverb’s markup)
    return s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n") \
            .replace("</p>", "\n").replace("<p>", "").replace("<b>", "").replace("</b>", "") \
            .replace("<i>", "").replace("</i>", "").strip()

#CYCLES============================================
def process_page(listings: list, product_type, category):
    """Returns the page listings as a list of key-value dictionaries"""
    rows = []
    for listing in listings:
        row = filter_listing(listing, product_type, category)
        rows.append(row)

    return rows

#SAVING FUNCTIONS=====================================

def save_page_listings(listings, category_live_ids, unseen_ids, category_cache, category_file_path: str, counters, product_type, category):
    new_listings = []

    #parquet
    for listing in listings:
        if listing['id'] in category_live_ids:
            # Compare with last known version
            if has_meaningful_changes(listing, category_cache):
                #close previous listing
                close_existing_record(listing['id'], product_type, category, category_file_path)
                counters['closed'] += 1
                #and save new snapshot
                listing.update({
                "snap_valid_from": snapshot_ts,
                "snap_valid_to": None,
                "snap_is_current": True
                })
                new_listings.append(listing)
                counters['updated'] += 1
            else:
                counters['unchanged'] += 1
            # Remove from existing_live_ids (mark as "still active")
            unseen_ids.discard(listing['id'])
            pass
        else:
            listing.update({
                "snap_valid_from": snapshot_ts,
                "snap_valid_to": None,
                "snap_is_current": True
            })
            new_listings.append(listing)
            counters['new'] += 1
    
    if new_listings:
        save_new_listings_batch(new_listings, category_file_path)   

def save_new_listings_batch(listings, file_path):
    """Save multiple listings at once"""
    if not listings:
        return
        
    df = pd.DataFrame(listings)

    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)

        if existing_df.empty:
            #empty dataset: just save into it
            df.to_parquet(file_path, index=False)
        elif df.empty:
            #no data to save
            return    
        else:
            #concat and save
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(file_path, index=False)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path, index=False)
    
    logger.debug(f"Saved {len(listings)} new listings")

def has_meaningful_changes(listing, category_cache):
    changed = False
    record = category_cache.get(listing['id'], {})

    current_price = listing.get('price', '')
    current_buyer_price = listing.get('buyer_price', '')
    current_state = listing.get('state', '')

    price = record.get('price', '')
    buyer_price = record.get('buyer_price', '')
    state = record.get('state', '')

    PRICE_TOLERANCE = 2.0

    if abs(float(current_price or 0) - float(price or 0)) > PRICE_TOLERANCE:
        changed = True
    if abs(float(current_buyer_price or 0) - float(buyer_price or 0)) > PRICE_TOLERANCE:
        changed = True
    if current_state != state:
        changed = True

    return changed

def close_existing_record(listing_id: int, product_type: str, category: str, file_path: str):
    """
    Close the current version of an existing record by setting 
    snap_valid_to and snap_is_current = False before saving the new version
    """
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} doesn't exist, nothing to close")
        return
    
    # Load the parquet file
    df = pd.read_parquet(file_path)
    
    # Find the current record for this specific listing
    mask = (
        (df['id'] == listing_id) &
        (df['product_type'] == product_type) & 
        (df['category'] == category) &
        (df['snap_is_current'] == True)
    )
    
    # Close the existing record
    df.loc[mask, 'snap_valid_to'] = snapshot_ts
    df.loc[mask, 'snap_is_current'] = False
    
    # Save back to file
    df.to_parquet(file_path, index=False)
    
    closed_count = mask.sum()
    logger.debug(f"Closed {closed_count} existing record for listing ID {listing_id}")

def close_unseen_listings(unseen_ids: set, product_type: str, category: str, file_path: str):
    """
    Close listings that weren't found in the current scan by setting 
    snap_valid_to and snap_is_current = False
    """
    if not unseen_ids:
        logger.info(f"No unseen listings to close for {product_type}|{category}")
        return 0
    
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} doesn't exist, nothing to close")
        return 0
    
    # Load the parquet file
    df = pd.read_parquet(file_path)
    
    # Filter for records that need to be closed
    mask = (
        (df['product_type'] == product_type) & 
        (df['category'] == category) &
        (df['snap_is_current'] == True) &
        (df['id'].isin(unseen_ids))
    )
    
    # Update the records
    df.loc[mask, 'snap_valid_to'] = snapshot_ts
    df.loc[mask, 'snap_is_current'] = False
    
    # Save back to file
    df.to_parquet(file_path, index=False)
    
    closed_count = mask.sum()
    logger.info(f"Closed {closed_count} unseen listings for {product_type}|{category}")
    return closed_count

#loading functions
def load_category_cache(product_type: str, category: str, file_path): #modify: should get correct file path
    if not os.path.exists(file_path):
        logger.critical(f"Failed Live listings loading for {product_type}|{category} at {file_path}")
        return {}
    else:
        logger.debug(f"Read previous listings at {file_path}")
    
    df = pd.read_parquet(file_path)

    filtered = df[(df['product_type'] == product_type) & 
                     (df['category'] == category) & 
                     (df['snap_is_current'] == True)]

    records_cache = {}

    for _, row in filtered.iterrows():
        records_cache[row['id']] = row.to_dict()
    
    return records_cache

def get_file_path(product_type):
    cohort = condition + "-" + HEADERS.get('X-Shipping-Region', '')
    folder = "test" if TEST_MODE == True else 'prod'
    return f"data/reverb/{folder}/{cohort}/{product_type}.parquet" #CHANGE WHEN EXITING TEST MODE

#EXECUTION FUNCTIONS==================================================

#will be in a cycle
def scan_category(href, product_type, category, PAGE_LIMIT = 10000, MAX_RETRIES = 25):

    category_file_path = get_file_path(product_type)

    #load a cache of live listings for current category, for performance when comparing changes, and extracts a set of live ids
    category_cache = load_category_cache(product_type, category, category_file_path)
    category_live_ids = set(category_cache.keys())

    counters = {'new': 0, 'updated': 0, 'unchanged': 0, 'closed': 0}

    unseen_ids = category_live_ids.copy()

    url = BASE+href
    logger.info(f"STARTING SCAN OF CATEGORY: {product_type}|{category} at {href}")

    last_observed_time = to_utc("2010-04-19T14:03:11-05:00") #load from previous session metadata? from the DB?

    next_link = None
    last_created_time = None
    #first request

    session = make_session()

    ok = False
    landing_attempts = 0
    max_landing_attempts = 10
    while not ok and landing_attempts < max_landing_attempts:
        try:
            response = session.get(url)
            data_page = response.json()
            ok = True
        except Exception as e:
            logger.error(f"Error {e} with initial request for {product_type}|{category}. Attempt {landing_attempts}.")
            landing_attempts += 1
            time.sleep(3)

    if landing_attempts == max_landing_attempts:
        logger.critical(f"Attention: Couldn't reach {product_type}|{category}. Skipping")
        return 0

    if data_page['listings']:
        last_created_time = to_utc(data_page['listings'][0]['created_at'])
        next_link = data_page.get('_links', {}).get('next', {}).get('href', None)

    current_page = data_page['current_page']
    total_pages = data_page['total_pages']

    #RETRY SETUP
    retry_count = 0

    logger.debug(f"Entering while cycle with next_link = {next_link}")
    logger.debug(f"Last seen in session: {last_created_time} | for category: {last_observed_time}")
    logger.info(f"Starting processing of {product_type}|{category}. {total_pages} pages found.")

    #SCANNING PHASE
    while True:
        
        if last_created_time and last_created_time < last_observed_time:
            logger.info("Found old listings. Exiting category")
            break
        if retry_count > MAX_RETRIES:
            logger.error("Too many retries. Exiting category")
            break
        if current_page > PAGE_LIMIT:
            logger.error(f"Beyond set page limit {PAGE_LIMIT}")
            break
        if current_page > total_pages:
            logger.error(f"Gone beyond initial pages found {total_pages}")
            break

        #incapsulate in an if to check if they exist

        #read response fields
        current_page = data_page['current_page']
        listings = data_page['listings']

        #log progress
        if (current_page-1) % 5 == 0: 
            logger.info(f"Pages [{current_page}/{total_pages}]")
        
        #save functions
        processed_listings = process_page(listings, product_type, category)
        save_page_listings(processed_listings, category_live_ids, unseen_ids, category_cache, category_file_path, counters, product_type, category)
        
        logger.debug(f"Saving phase completed for PAGE {current_page} (of {total_pages})")

        #last page checkers. BEFORE the request.
        if data_page['_links']:
            next_link = data_page.get('_links', {}).get('next', {}).get('href', None)

            if next_link:
                next_link = re.sub(r'per_page=\d+', 'per_page=50', next_link)
            else:
                logger.info("Last Page Reached")
                break
        else: 
            next_link = None
            logger.info("Links not found. Assuming last page.")
            break

        try:
            time.sleep(random.uniform(0.08, 0.15))
            response = session.get(next_link) #modify
            logger.debug(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data_page = response.json()
                retry_count = 0
                logger.debug("Request OK")
            elif response.status_code == 429:
                logger.error(f"Beware of 429. Retry count: {retry_count}")
                time.sleep(random.uniform(3, 5))
                retry_count += 1
                continue
            else: 
                logger.error(f"HTTP {response.status_code} error at retry {retry_count}")
                retry_count += 1
                continue
            
        except Exception as e:
            logger.error(f"requests Error: {e}")
            continue
        
        if data_page['listings']:
            last_created_time = to_utc(data_page['listings'][0]['created_at']) #update last seen time

    session.close()

    logger.info(f"Save position: {category_file_path}")
    logger.info(f"Category Recap {"=" * 15}* \n \
                 New: {counters['new']}, Updated: {counters['updated']}, Unchanged: {counters['unchanged']}")
    
    if current_page == total_pages:
        logger.info(f"Succesfully Completed FULL SCAN of {product_type}|{category}")

        closed_listings = close_unseen_listings(unseen_ids, product_type, category, category_file_path)

        counters['closed'] += closed_listings

        return {
            "status": "completed", 
            "counters": counters
        }
    
    elif current_page < total_pages:
        logger.warning(f"Didn't reach end of scan for {product_type}|{category}. Scanned: {current_page} Remaining: {total_pages-current_page}")
        return {
            "status": "partial", 
            "counters": counters
        }
    else:
        return {"status": "completed", "counters": counters}
                

#TEST MODE CONFIGURATION
TEST_MODE = True
PARTIAL_MODE = False
PARTIAL_PRODUCT_TYPES = ['bass-guitars']

if __name__ == "__main__":
    cat_table = pd.read_csv("data/reverb/meta/product_type-cat-href_table.csv")

    # Global session counters 
    global_counters = {'new': 0, 'updated': 0, 'unchanged': 0, 'closed': 0}
    session_summary = {'completed': 0, 'partial': 0, 'failed': 0}

    if PARTIAL_MODE:
        available_types = set(cat_table['product_type'].unique())

        for el in PARTIAL_PRODUCT_TYPES:
            if el not in available_types:
                raise ValueError(f"{el} product_type selection not found. Stopping execution.")

        product_types = PARTIAL_PRODUCT_TYPES
        logger.info(f"PARTIAL Mode Set: Processing only {PARTIAL_PRODUCT_TYPES}")

    else:
        product_types = cat_table['product_type'].unique()
        logger.info("COMPLETE Mode Set: Processing all product types")
    
    for product_type in product_types:
        logger.info(f"Starting product type: {product_type}")
        
        # Filter rows for this product type
        product_rows = cat_table[cat_table['product_type'] == product_type]
        
        for _, row in product_rows.iterrows():
            category = row['category']
            href = row['listings_href']  
            try:
                logger.info(f"Processing {product_type}|{category}")

                scan_result = scan_category(href, product_type, category)

                if scan_result == 0:
                    logger.error(f"Could not reach {product_type}|{category}")
                    session_summary['failed'] += 1
                    continue

                logger.info(f"Finished {product_type}|{category} with status: {scan_result['status']}")
                
                #update global counters
                for key in global_counters:
                    global_counters[key] += scan_result['counters'][key]
                
                session_summary[scan_result['status']] += 1

                # Rate limiting between categories
                wait_time = random.uniform(3, 8)
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Failed to process {product_type}|{category}: {e}")
                session_summary['failed'] += 1  # Add this line
                continue
    
    logger.info("=" * 50)
    logger.info("SESSION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Categories: Completed={session_summary['completed']}, Partial={session_summary['partial']}, Failed={session_summary['failed']}")
    logger.info(f"Total Changes: New={global_counters['new']}, Updated={global_counters['updated']}, Unchanged={global_counters['unchanged']}, Closed={global_counters['closed']}")
    logger.info(f"Log saved at: {log_filename}")