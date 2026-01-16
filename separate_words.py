import pandas as pd
import csv


# 1. Load your dataset
df = pd.read_csv('generated_product_types.csv')

def check_separation(text):
    if pd.isna(text):
        return False
    text_str = str(text)
    
    # Condition 1: Word count > 2
    word_count = len(text_str.split())
    
    # Condition 2: Contains double quotes already
    has_quotes = '"' in text_str
    
    return word_count > 3 or has_quotes

# 2. Apply the filtering logic
mask = df['Generated_Product_Type'].apply(check_separation)

df_more_than_two = df[mask].copy()
df_others = df[~mask].copy()

# 3. Save files
# QUOTE_MINIMAL only adds quotes if the data contains a comma or existing quotes
df_more_than_two.to_csv('more_than_two_words.csv', index=False, quoting=csv.QUOTE_MINIMAL)
df_others.to_csv('other_products.csv', index=False, quoting=csv.QUOTE_MINIMAL)

print("Files generated successfully.")