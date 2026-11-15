
import sqlite3
from pathlib import Path

# Find the database file
folder = Path(__file__).resolve().parent
db_file = folder / "books.db"

# Connect to the database
connection = sqlite3.connect(db_file)
cursor = connection.cursor()

# Store SQL queries and their descriptions
queries = [
    (
        "Query 1: Books with price greater than £40",
        """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp > 40
        """
    ),
    (
        "Query 2: Books sorted by price (highest first)",
        """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        """
    ),
    (
        "Query 3: First 10 books",
        """
        SELECT title, rating
        FROM books
        LIMIT 10
        """
    ),
    (
        "Query 4: List of distinct categories",
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
        """
    ),
    (
        "Query 5: Books rated 4 or 5",
        """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        """
    ),
    (
        "Query 6: Books with category names (JOIN)",
        """
        SELECT books.title, books.price_gbp,
               categories.category_name
        FROM books
        JOIN categories
        ON books.category_id = categories.category_id
        ORDER BY books.title
        """
    )
]

# Save query descriptions and results
output_file = folder / "sql_query_results.txt"

with open(output_file, "w", encoding="utf-8") as file:
    for description, sql in queries:
        file.write(description + "\n")
        file.write("=" * 60 + "\n")
        print("\n" + "=" * 60)
        print(description)
        print("=" * 60)

        cursor.execute(sql)
        results = cursor.fetchall()

        for row in results:
            print(row)
            file.write(str(row) + "\n")

        # Add query heading to the saved results
        # (The heading is also displayed in the terminal.)
        file.write("\n")

        # Keep the description in the output file
        # on the next run's corresponding section.
        
connection.close()

print("\nAll SQL queries executed!")
print("Results saved to:", output_file)
