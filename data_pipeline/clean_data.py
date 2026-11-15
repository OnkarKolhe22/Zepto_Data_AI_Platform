
import pandas as pd
from pathlib import Path

# Find the data_pipeline folder
folder = Path(__file__).resolve().parent

# Read the raw scraped data
input_file = folder / "books_raw.csv"

df = pd.read_csv(input_file)

print("Original data shape:", df.shape)

# --------------------------------
# 1. Clean book prices
# --------------------------------

# Remove currency symbols and extract numeric values
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"].astype(str).str.extract(r"([\d.]+)")[0],
    errors="coerce"
)

# --------------------------------
# 2. Clean star ratings
# --------------------------------

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)

# --------------------------------
# 3. Handle missing numeric values
# --------------------------------

# Use median imputation for missing or invalid prices
if df["price_gbp"].notna().any():
    df["price_gbp"] = df["price_gbp"].fillna(
        df["price_gbp"].median()
    )
else:
    raise ValueError("All book prices are missing or invalid.")

# Use median imputation for missing or invalid ratings
if df["rating"].notna().any():
    median_rating = df["rating"].median()
    df["rating"] = df["rating"].fillna(
        round(median_rating)
    ).round().astype(int)
else:
    raise ValueError("All book ratings are missing or invalid.")

# --------------------------------
# 4. Create stock status
# --------------------------------

df["in_stock"] = df["availability"].astype(str).str.contains(
    r"\bIn stock\b",
    case=False,
    na=False
)

# --------------------------------
# 5. Convert GBP to INR
# --------------------------------

# Fixed exchange rate from assignment
EXCHANGE_RATE = 105.50

df["price_inr"] = (
    df["price_gbp"] * EXCHANGE_RATE
).round(2)

# --------------------------------
# 6. Remove rows missing essential text
# --------------------------------

before = len(df)

df = df.dropna(subset=["title", "category"]).copy()

removed = before - len(df)

print("Rows removed due to missing title/category:", removed)

# --------------------------------
# 7. Select final columns
# --------------------------------

clean_df = df[
    [
        "title",
        "price_gbp",
        "rating",
        "availability",
        "in_stock",
        "category",
        "price_inr"
    ]
].copy()

# --------------------------------
# 8. Save cleaned data
# --------------------------------

output_file = folder / "books_clean.csv"

clean_df.to_csv(output_file, index=False)

# --------------------------------
# 9. Display results
# --------------------------------

print("\nData cleaning completed!")

print("\nCleaned data shape:")
print(clean_df.shape)

print("\nCleaned data types:")
print(clean_df.dtypes)

print("\nFirst five cleaned books:")
print(clean_df.head())

print("\nMissing values:")
print(clean_df.isnull().sum())

print("\nSaved cleaned data to:", output_file)
