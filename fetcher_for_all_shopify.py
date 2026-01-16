import os
import requests
import pandas as pd
import subprocess
import re
import time
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================= USER CONFIG =================
# Set MODE to "ALL" to process the entire store
# Set MODE to "COLLECTION" to process specific handles below
MODE = "ALL"  

# Only used if MODE = "COLLECTION"
COLLECTION_HANDLES = ["garden-outdoors", "parasol"] 

CSV_OUTPUT = "generated_product_types.csv"
# ===============================================

load_dotenv()

STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
API_VERSION = os.getenv("API_VERSION", "2024-01")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

STORE_URL = f"{STORE_NAME}.myshopify.com"
HEADERS = {"X-Shopify-Access-Token": ACCESS_TOKEN, "Content-Type": "application/json"}

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --- Helper Functions (Ollama, Captioning, Normalization) ---

def caption_image(image_url):
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_length=40)
        return processor.decode(output[0], skip_special_tokens=True)
    except: return None

def ollama_query(prompt):
    try:
        result = subprocess.run(["ollama", "run", "mistral"], input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return result.stdout.decode("utf-8", errors="ignore").strip()
    except: return ""

def generate_product_type(caption):
    prompt = f"Identify the main product. Rules: 1-3 words, singular noun, no color. Description: '{caption}'"
    raw = ollama_query(prompt)
    text = re.sub(r"\(.*?\)|['\"]", "", raw)
    text = re.split(r"assuming|however|note:", text, flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", text).strip().title()

def extract_color(caption):
    prompt = f"Identify the main color from this: '{caption}'. One word only."
    raw = ollama_query(prompt).lower()
    match = re.search(r"\b(black|white|grey|gray|beige|tan|brown|blue|green|red|pink|yellow|orange)\b", raw)
    return match.group(1).title() if match else ""

# --- Shopify Data Fetching ---

def fetch_products(url):
    all_products = []
    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        all_products.extend(response.json().get("products", []))
        # Handle Pagination (Link Header)
        link = response.headers.get("Link")
        url = re.search(r'<(https://[^>]+)>;\s*rel="next"', link).group(1) if link and 'rel="next"' in link else None
    return all_products

def get_collection_id(handle):
    for endpoint in ["custom_collections", "smart_collections"]:
        url = f"https://{STORE_URL}/admin/api/{API_VERSION}/{endpoint}.json"
        r = requests.get(url, headers=HEADERS)
        for col in r.json().get(endpoint, []):
            if col["handle"] == handle: return col["id"]
    return None

# ================= MAIN =================

def main():
    print(f"🚀 MODE: {MODE}")
    
    # 1. Load existing progress to avoid re-processing
    processed_rows = []
    processed_ids = set()
    if os.path.exists(CSV_OUTPUT):
        existing_df = pd.read_csv(CSV_OUTPUT)
        processed_rows = existing_df.to_dict('records')
        processed_ids = set(existing_df['Product_ID'].astype(str))
        print(f"Loaded {len(processed_ids)} existing products from CSV. Resuming...")

    try:
        # 2. Collect Products
        all_to_process = []
        if MODE == "ALL":
            print("Fetching all products from store...")
            api_url = f"https://{STORE_URL}/admin/api/{API_VERSION}/products.json?limit=250"
            all_to_process = fetch_products(api_url)
        else:
            for handle in COLLECTION_HANDLES:
                coll_id = get_collection_id(handle)
                if coll_id:
                    api_url = f"https://{STORE_URL}/admin/api/{API_VERSION}/collections/{coll_id}/products.json?limit=250"
                    all_to_process.extend(fetch_products(api_url))

        # 3. Process
        total = len(all_to_process)
        print(f"Total products to check: {total}")

        for idx, product in enumerate(all_to_process, 1):
            p_id = str(product['id'])
            
            if p_id in processed_ids:
                continue # Skip already done
            
            print(f"[{idx}/{total}] Processing: {product['title'][:30]}...")

            images = product.get("images", [])
            if not images:
                processed_ids.add(p_id)
                continue

            caption = caption_image(images[0]["src"]) or product["title"]
            p_type = f"{generate_product_type(caption)} {extract_color(caption)}".strip()
            
            variants = product.get("variants", [])
            ref_sku = variants[0].get("sku") if variants else f"PROD-{p_id}"

            processed_rows.append({
                "Product_ID": p_id,
                "SKU": ref_sku,
                "Generated_Product_Type": p_type
            })
            processed_ids.add(p_id)

            # Auto-save every 10 products just in case
            if idx % 10 == 0 and not DRY_RUN:
                pd.DataFrame(processed_rows).to_csv(CSV_OUTPUT, index=False)

        # 4. Final Save
        if not DRY_RUN:
            pd.DataFrame(processed_rows).to_csv(CSV_OUTPUT, index=False)
            print(f"✅ Completed! Final file contains {len(processed_rows)} products.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()