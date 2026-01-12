import os
import csv
import time
import argparse
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

# ================= CONFIG =================
CSV_FILE = "generated_product_types.csv"
REQUEST_DELAY = 0.5     # seconds between requests
MAX_RETRIES = 3
# =========================================

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

# ---------- ENV ----------
load_dotenv()

STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
API_VERSION = os.getenv("API_VERSION")

DRY_RUN = args.dry_run or os.getenv("DRY_RUN", "false").lower() == "true"

BASE_URL = f"https://{STORE_NAME}.myshopify.com/admin/api/{API_VERSION}"
HEADERS = {
    "X-Shopify-Access-Token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

# ------------------------------------------------

def safe_get(url, params=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            r.raise_for_status()
            return r
        except RequestException as e:
            print(f"⚠️ GET failed (attempt {attempt}): {e}")
            time.sleep(attempt * 2)
    return None

def safe_put(url, payload):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.put(url, headers=HEADERS, json=payload, timeout=15)
            r.raise_for_status()
            return r
        except RequestException as e:
            print(f"⚠️ PUT failed (attempt {attempt}): {e}")
            time.sleep(attempt * 2)
    return None

# ------------------------------------------------

def find_variant_by_sku(sku):
    url = f"{BASE_URL}/variants.json"
    r = safe_get(url, params={"sku": sku})
    if not r:
        return None, None

    variants = r.json().get("variants", [])
    if not variants:
        return None, None

    v = variants[0]
    return v["id"], v["product_id"]

def update_product_type(product_id, product_type):
    url = f"{BASE_URL}/products/{product_id}.json"

    payload = {
        "product": {
            "id": product_id,
            "product_type": product_type
        }
    }

    if DRY_RUN:
        print(f"[DRY RUN] Would update product {product_id} → {product_type}")
        return True

    r = safe_put(url, payload)
    if r:
        print(f"✅ Updated product {product_id} → {product_type}")
        return True

    print(f"❌ Failed updating product {product_id}")
    return False

# ------------------------------------------------

print("\n========== SHOPIFY PRODUCT TYPE UPDATE ==========")
print("CSV:", CSV_FILE)
print("DRY_RUN:", DRY_RUN)
print("=================================================")

updated_products = set()
skipped = 0
failed = 0

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        sku = row["SKU"].strip()
        product_type = row["Generated_Product_Type"].strip()

        print("\n---------------------------------")
        print("SKU:", sku)
        print("Product Type:", product_type)

        variant_id, product_id = find_variant_by_sku(sku)

        if not product_id:
            print("⛔ SKU not found — skipped")
            skipped += 1
            continue

        if product_id in updated_products:
            print("↪ Product already updated — skipped")
            continue

        success = update_product_type(product_id, product_type)
        if success:
            updated_products.add(product_id)
        else:
            failed += 1

        time.sleep(REQUEST_DELAY)

print("\n=================================================")
print("TOTAL PRODUCTS UPDATED:", len(updated_products))
print("SKIPPED:", skipped)
print("FAILED:", failed)
print("=================================================")
