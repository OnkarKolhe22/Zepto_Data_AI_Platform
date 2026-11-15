
import sqlite3
import pandas as pd
from pathlib import Path

# Locate the database
folder = Path(__file__).resolve().parent
db_file = folder / "books.db"

# Connect to SQLite
connection = sqlite3.connect(db_file)

# Read the first SQL result into pandas
print("\n--- SQL Result 1: Books costing more than £40 ---")

expensive_books = pd.read_sql(
    """
    SELECT title, price_gbp
    FROM books
    WHERE price_gbp > 40
    """,
    connection
)

print(expensive_books.head())
print("Number of books costing more than £40:", len(expensive_books))


# Read the second SQL result into pandas
print("\n--- SQL Result 2: Books with category names ---")

sql_join = pd.read_sql(
    """
    SELECT
        books.title,
        books.price_gbp,
        books.rating,
        categories.category_name
    FROM books
    JOIN categories
    ON books.category_id = categories.category_id
    ORDER BY books.title
    """,
    connection
)

print(sql_join.head())


# Read the two database tables separately
books_df = pd.read_sql("SELECT * FROM books", connection)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    connection
)

# Reproduce the SQL JOIN using pandas merge
print("\n--- Pandas merge result ---")

pandas_join = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

# Select the same columns and sort in the same order
pandas_join = pandas_join[
    ["title", "price_gbp", "rating", "category_name"]
].sort_values("title").reset_index(drop=True)

sql_join = sql_join.reset_index(drop=True)

print(pandas_join.head())


# Compare both results
print("\n--- SQL JOIN vs Pandas merge comparison ---")

if sql_join.equals(pandas_join):
    print("SUCCESS: SQL JOIN and pandas merge produce identical results!")
else:
    print("The results are different. We need to investigate.")

connection.close()
