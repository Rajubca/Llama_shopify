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
MODE = "COLLECTION"  # "ALL" or "COLLECTION"
COLLECTION_HANDLE = ["garden-outdoors", "parasol"]
CSV_OUTPUT = "generated_product_types.csv"
# ===============================================

load_dotenv()

STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
API_VERSION = os.getenv("API_VERSION", "2024-01")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

STORE_URL = f"{STORE_NAME}.myshopify.com"
HEADERS = {"X-Shopify-Access-Token": ACCESS_TOKEN, "Content-Type": "application/json"}

# Load BLIP model once
logging_level = "ERROR" # Reduce HF noise
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image_url):
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_length=40)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"   ⚠️ Captioning failed: {e}")
        return None

def ollama_query(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        return result.stdout.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"   ⚠️ Ollama error: {e}")
        return ""

def generate_product_type(caption):
    prompt = f"Identify the main product. Rules: 1-3 words, singular noun, no color/brand/size. Description: '{caption}'. Return only the product type."
    raw_type = ollama_query(prompt)
    return normalize_text(raw_type)

def extract_color(caption):
    prompt = f"Identify the main color from this description: '{caption}'. Return one word only or empty if unclear."
    raw_color = ollama_query(prompt)
    match = re.search(r"\b(black|white|grey|gray|beige|tan|brown|blue|green|red|pink|yellow|orange)\b", raw_color.lower())
    return match.group(1).title() if match else ""

def normalize_text(text):
    text = re.sub(r"\(.*?\)|['\"]", "", text)
    text = re.split(r"assuming|however|note:|based on", text, flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", text).strip().title()

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
    raise ValueError(f"Collection {handle} not found.")

# ================= MAIN =================

def main():
    print(f"🚀 Starting Generation (Mode: {MODE})")
    
    try:
        if MODE == "COLLECTION":
            coll_id = get_collection_id(COLLECTION_HANDLE)
            api_url = f"https://{STORE_URL}/admin/api/{API_VERSION}/collections/{coll_id}/products.json?limit=250"
        else:
            api_url = f"https://{STORE_URL}/admin/api/{API_VERSION}/products.json?limit=250"

        products = fetch_products(api_url)
        print(f"Found {len(products)} products.")

        rows = []
        for idx, product in enumerate(products, 1):
            p_id = product['id']
            print(f"[{idx}/{len(products)}] Processing: {product['title'][:40]}...")

            images = product.get("images", [])
            if not images:
                print("   ⏩ Skip: No image")
                continue

            caption = caption_image(images[0]["src"]) or product["title"]
            base_type = generate_product_type(caption)
            color = extract_color(caption)
            
            final_type = f"{base_type} {color}".strip()
            
            # Use first SKU as a reference, but we primarily care about Product ID
            variants = product.get("variants", [])
            ref_sku = variants[0].get("sku") if variants else f"PROD-{p_id}"

            # WE ONLY ADD ONE ROW PER PRODUCT
            rows.append({
                "Product_ID": p_id,
                "SKU": ref_sku,
                "Generated_Product_Type": final_type
            })

        df = pd.DataFrame(rows)
        if not df.empty and not DRY_RUN:
            df.to_csv(CSV_OUTPUT, index=False)
            print(f"✅ Created {CSV_OUTPUT} with {len(rows)} unique products.")
        else:
            print("Preview of data:")
            print(df.head())

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    main()