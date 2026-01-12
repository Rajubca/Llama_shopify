import os
import csv
import time
import logging
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

# ---------- SETUP LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

load_dotenv()

CSV_FILE = "generated_product_types.csv"
STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
API_VERSION = os.getenv("API_VERSION", "2024-01")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

BASE_URL = f"https://{STORE_NAME}.myshopify.com/admin/api/{API_VERSION}"

# ---------- PERSISTENT SESSION ----------
session = requests.Session()
session.headers.update({
    "X-Shopify-Access-Token": ACCESS_TOKEN,
    "Content-Type": "application/json"
})

def update_product_type(product_id, new_type):
    if DRY_RUN:
        logging.info(f"[DRY RUN] Would update {product_id} -> {new_type}")
        return True

    url = f"{BASE_URL}/products/{product_id}.json"
    payload = {"product": {"id": product_id, "product_type": new_type}}
    
    try:
        r = session.put(url, json=payload, timeout=15)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2))
            logging.warning(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            return update_product_type(product_id, new_type)
        
        r.raise_for_status()
        logging.info(f"SUCCESS: Product {product_id} updated to '{new_type}'")
        return True
    except Exception as e:
        logging.error(f"FAILED: Product {product_id}: {e}")
        return False

def main():
    if not os.path.exists(CSV_FILE):
        logging.error(f"CSV file {CSV_FILE} not found!")
        return

    success_count = 0
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_id = row.get("Product_ID")
            p_type = row.get("Generated_Product_Type")

            if p_id and p_type:
                if update_product_type(p_id, p_type):
                    success_count += 1
                time.sleep(0.1) # Small delay to be polite

    logging.info(f"Done. Total Products Updated: {success_count}")

if __name__ == "__main__":
    main()