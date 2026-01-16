import os
import requests
import pandas as pd
import re
import base64
from dotenv import load_dotenv
from openai import OpenAI

# ================= USER CONFIG =================
MODE = "ALL"  # "ALL" or "COLLECTION"
COLLECTION_HANDLES = ["garden-outdoors"] 
CSV_OUTPUT = "generated_product_types.csv"
# ===============================================

load_dotenv()

# API Keys
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
API_VERSION = os.getenv("API_VERSION", "2024-01")

STORE_URL = f"{STORE_NAME}.myshopify.com"
HEADERS = {"X-Shopify-Access-Token": ACCESS_TOKEN, "Content-Type": "application/json"}

def analyze_image_with_gpt(image_url):
    """Uses GPT-4o to identify product type and color in one go."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify this product. Return ONLY: [Singular Product Name] [Color]. Use 1-3 words for the product. No brand names. Example: 'Patio Chair Black'"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=20,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ⚠️ OpenAI Error: {e}")
        return None

# --- Shopify Data Fetching ---

def fetch_products(url):
    all_products = []
    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        all_products.extend(response.json().get("products", []))
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
    print(f"🚀 MODE: {MODE} (Using OpenAI GPT-4o)")
    
    # 1. Resume Progress
    processed_rows = []
    processed_ids = set()
    if os.path.exists(CSV_OUTPUT):
        existing_df = pd.read_csv(CSV_OUTPUT)
        processed_rows = existing_df.to_dict('records')
        processed_ids = set(existing_df['Product_ID'].astype(str))
        print(f"Loaded {len(processed_ids)} existing products. Resuming...")

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

        total = len(all_to_process)
        print(f"Total products to check: {total}")

        # 3. Process with OpenAI
        for idx, product in enumerate(all_to_process, 1):
            p_id = str(product['id'])
            if p_id in processed_ids: continue
            
            print(f"[{idx}/{total}] AI Analyzing: {product['title'][:30]}...")

            images = product.get("images", [])
            if not images:
                processed_ids.add(p_id)
                continue

            # GPT-4o analyzes the direct image URL
            p_type = analyze_image_with_gpt(images[0]["src"])
            
            if not p_type: 
                p_type = "Unknown" # Fallback

            variants = product.get("variants", [])
            ref_sku = variants[0].get("sku") if variants else f"PROD-{p_id}"

            processed_rows.append({
                "Product_ID": p_id,
                "SKU": ref_sku,
                "Generated_Product_Type": p_type
            })
            processed_ids.add(p_id)

            # Save progress frequently
            if idx % 5 == 0:
                pd.DataFrame(processed_rows).to_csv(CSV_OUTPUT, index=False)

        # 4. Final Save
        pd.DataFrame(processed_rows).to_csv(CSV_OUTPUT, index=False)
        print(f"✅ Completed! Results saved to {CSV_OUTPUT}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()