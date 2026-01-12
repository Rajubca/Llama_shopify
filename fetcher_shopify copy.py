import os
import requests
import pandas as pd
import subprocess
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================= USER CONFIG =================
# MODE = "ALL"  # "ALL" or "COLLECTION"
MODE =  "COLLECTION"
COLLECTION_HANDLE = "parasol"  # required only if MODE = "COLLECTION"
COLLECTION_HANDLE = "garden-outdoors"  # required only if MODE = "COLLECTION"
SPECIAL_COLLECTION_RULES = {
    "parasols": "PARASOL_STAND_COLOR_SIZE",
    "chairs": "CHAIR_MATERIAL_STYLE",
}

CANONICAL_PRODUCT_TYPES = {
    "heater": "Outdoor Heater",
    "radiant heater": "Outdoor Heater",
    "heater lamp": "Outdoor Heater",

    "furniture set": "Outdoor Furniture Set",
    "outdoor furniture": "Outdoor Furniture Set",
    "wicker set": "Outdoor Furniture Set",
    "wicker furniture": "Outdoor Furniture Set",

    "chair": "Garden Chair",
    "garden chair": "Garden Chair",
    "patio chair": "Garden Chair",

    "electric insect trap": "Electric Insect Trap",
    "insect trap": "Electric Insect Trap",

    "paper butterfly": "Paper Decoration",
    "paper decor": "Paper Decoration",
    "paper decoration": "Paper Decoration",

    "fan": "Electric Fan",
    "fan pack": "Electric Fan",

    "badminton racket": "Badminton Racket",

    "sofa": "Outdoor Sofa",
    "table": "Garden Table",
}

# ===============================================

# Load environment
load_dotenv()

STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
API_VERSION = os.getenv("API_VERSION")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

STORE_URL = f"{STORE_NAME}.myshopify.com"

HEADERS = {
    "X-Shopify-Access-Token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

# Load BLIP model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=False
)

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image_url, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(image_url, timeout=20)
            resp.raise_for_status()

            image = Image.open(BytesIO(resp.content)).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs, max_length=40)
            return processor.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            print(f"⚠️ Image fetch failed (attempt {attempt}): {e}")

    # Final fallback
    print("❌ Image skipped after retries")
    return None

def generate_product_type(caption):
    prompt = f"""
Generate a generic product type name from the image description below.

Strict rules:
- Use 2 to 4 words only
- Use singular noun form
- No colors
- No brand names
- No marketing adjectives
- No punctuation

Image description:
"{caption}"

Return only the product type.
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode("utf-8", errors="ignore").strip()

def fetch_products_all():
    products = []
    url = f"https://{STORE_URL}/admin/api/{API_VERSION}/products.json?limit=250"

    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        products.extend(data.get("products", []))

        link = response.headers.get("Link")
        if link and 'rel="next"' in link:
            url = link.split(";")[0].strip("<>")
        else:
            url = None

    return products

def fetch_products_collection(handle):
    collection_id = get_collection_id_by_handle(handle)

    products = []
    url = f"https://{STORE_URL}/admin/api/{API_VERSION}/collections/{collection_id}/products.json?limit=250"

    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        products.extend(data.get("products", []))

        link = response.headers.get("Link")
        if link and 'rel="next"' in link:
            url = link.split(";")[0].strip("<>")
        else:
            url = None

    return products

def get_collection_id_by_handle(handle):
    # Check custom collections
    url = f"https://{STORE_URL}/admin/api/{API_VERSION}/custom_collections.json?limit=250"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    for col in r.json().get("custom_collections", []):
        if col["handle"] == handle:
            return col["id"]

    # Check smart collections
    url = f"https://{STORE_URL}/admin/api/{API_VERSION}/smart_collections.json?limit=250"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    for col in r.json().get("smart_collections", []):
        if col["handle"] == handle:
            return col["id"]

    raise ValueError(f"Collection handle not found: {handle}")

def extract_color(caption):
    prompt = f"""
Identify the main visible color of the product in the image.

Rules:
- Mention the color plainly
- Avoid explanations if possible

Image description:
"{caption}"
"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    raw = result.stdout.decode("utf-8", errors="ignore").strip().lower()

    # --- CLEANING STEP (THIS IS THE KEY FIX) ---
    # Extract first reasonable color word from the response
    match = re.search(
        r"\b(black|white|grey|gray|beige|tan|brown|blue|green|red|pink)\b",
        raw
    )

    if match:
        return match.group(1)

    # Fallback: take first word only (last safety net)
    return raw.split()[0] if raw else "black"

def extract_size_from_title(title):
    patterns = [
        r"\d+\s?kg",           # 12kg, 35 kg
        r"\d+(\.\d+)?m",       # 3m, 3.5m
        r"\d+x\d+m",           # 3x3m
        r"\d+x\d+x\d+cm",      # 103.5x103.5x7.5CM
        r"\d+mm",              # 38mm
    ]

    for p in patterns:
        match = re.search(p, title.lower())
        if match:
            return match.group().upper()

    return ""

def generate_product_type_by_rule(rule, product, caption):
    if rule == "PARASOL_STAND_COLOR_SIZE":
        return rule_parasol_stand(product, caption)

    if rule == "CHAIR_MATERIAL_STYLE":
        return rule_chair_material_style(product, caption)

    # fallback safety
    return generate_product_type(caption)

def rule_parasol_stand(product, caption):
    color = extract_color(caption)
    size = extract_size_from_title(product["title"])

    if size:
        return f"{color.title()} Parasol Stand {size}"
    else:
        return f"{color.title()} Parasol Stand"

def extract_material_from_title(title):
    materials = [
        "wood", "wooden",
        "metal", "steel",
        "plastic",
        "rattan",
        "fabric"
    ]

    title_lower = title.lower()
    for m in materials:
        if m in title_lower:
            return m.capitalize()

    return "Generic"

def rule_chair_material_style(product, caption):
    material = extract_material_from_title(product["title"])

    if "garden" in caption.lower() or "outdoor" in caption.lower():
        return f"{material} Garden Chair"

    return f"{material} Chair"

def normalize_product_type(text):
    text = text.lower()

    # remove bracketed text
    text = re.sub(r"\(.*?\)", "", text)

    # remove explanatory phrases
    text = re.split(
        r"assuming|however|note:|if it|based on|without|single-word|singular",
        text
    )[0]

    # remove quotes
    text = text.replace('"', "").replace("'", "")

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # title case
    text = text.title()

    return text

def map_to_canonical_type(text):
    text_lower = text.lower()

    for key, canonical in CANONICAL_PRODUCT_TYPES.items():
        if key in text_lower:
            return canonical

    return text  # fallback if no match

# ================= MAIN (DEBUG MODE) =================

print("========== DEBUG START ==========")
print("MODE:", MODE)
print("COLLECTION_HANDLE:", COLLECTION_HANDLE)
print("STORE_URL:", STORE_URL)
print("API_VERSION:", API_VERSION)
print("DRY_RUN:", DRY_RUN)
print("=================================")

if MODE == "COLLECTION":
    products = fetch_products_collection(COLLECTION_HANDLE)
else:
    products = fetch_products_all()

print(f"TOTAL PRODUCTS FETCHED: {len(products)}")

rows = []

for idx, product in enumerate(products, start=1):
    print("\n---------------------------------")
    print(f"PRODUCT #{idx}")
    print("Product ID:", product.get("id"))
    print("Title:", product.get("title"))

    images = product.get("images", [])
    variants = product.get("variants", [])

    print("Image count:", len(images))
    print("Variant count:", len(variants))

    if not images:
        print("⛔ SKIPPED: No product images")
        continue

    main_image_url = images[0]["src"]
    print("Main image URL:", main_image_url)

    caption = caption_image(main_image_url)

    if not caption:
        # fallback: use title only
        caption = product["title"]

    print("Generated caption:", caption)

    rule = SPECIAL_COLLECTION_RULES.get(COLLECTION_HANDLE)

    if rule:
        product_type = generate_product_type_by_rule(rule, product, caption)
    else:
        # Generic logic for collections like garden-outdoors
        raw_type = generate_product_type(caption)
        clean_type = normalize_product_type(raw_type)
        product_type = map_to_canonical_type(clean_type)



    print("Generated product type:", product_type)

    if not variants:
        fallback_sku = f"PRODUCT-{product['id']}"
        rows.append({
            "SKU": fallback_sku,
            "Generated_Product_Type": product_type
        })
        print("  ✅ ROW ADDED (product-level fallback SKU)")
        continue


    for variant in variants:
        sku = variant.get("sku") or f"VARIANT-{variant['id']}"

        variant_id = variant.get("id")

        print("  Variant ID:", variant_id, "SKU:", sku)

        if not sku:
            print("  ⚠️ SKU missing — skipping variant")
            continue

        rows.append({
            "SKU": sku,
            "Generated_Product_Type": product_type
        })

        print("  ✅ ROW ADDED")

print("\n=================================")
print("TOTAL ROWS COLLECTED:", len(rows))
print("=================================")

df = pd.DataFrame(rows)

if df.empty:
    print("❌ DATAFRAME IS EMPTY — NO ROWS TO WRITE")
else:
    print("✅ DATAFRAME READY")

if DRY_RUN:
    print(df.head(10))
    print("DRY_RUN enabled — CSV not written.")
else:
    df.to_csv("generated_product_types.csv", index=False)
    print("CSV generated: generated_product_types.csv")
