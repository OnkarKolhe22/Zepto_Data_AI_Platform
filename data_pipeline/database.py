
import sqlite3
import pandas as pd
from pathlib import Path

# Locate the data_pipeline folder
folder = Path(__file__).resolve().parent

# File locations
csv_file = folder / "books_clean.csv"
db_file = folder / "books.db"

# Read cleaned book data
df = pd.read_csv(csv_file)

# Connect to SQLite database
connection = sqlite3.connect(db_file)

# Enable foreign key checking
connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()

# Create categories table
cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")

# Create books table
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    rating INTEGER NOT NULL,
    availability TEXT,
    in_stock INTEGER NOT NULL,
    price_inr REAL NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

# Insert unique categories
categories = df["category"].drop_duplicates()

for category in categories:
    cursor.execute(
        """
        INSERT OR IGNORE INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )

# Insert books with their category IDs
for _, row in df.iterrows():

    cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            rating,
            availability,
            in_stock,
            price_inr,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            int(row["rating"]),
            row["availability"],
            int(row["in_stock"]),
            row["price_inr"],
            category_id
        )
    )

# Save all database changes
connection.commit()

# Display database information
print("Database created successfully!")

cursor.execute("SELECT COUNT(*) FROM books")
book_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM categories")
category_count = cursor.fetchone()[0]

print("Total books:", book_count)
print("Total categories:", category_count)

# Display first five books
print("\nFirst five books from database:")

result = pd.read_sql_query(
    """
    SELECT
        books.title,
        books.price_gbp,
        books.rating,
        categories.category_name
    FROM books
    JOIN categories
    ON books.category_id = categories.category_id
    LIMIT 5
    """,
    connection
)

print(result)

connection.close()

print("\nDatabase saved at:", db_file)
