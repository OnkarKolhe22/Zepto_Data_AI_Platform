# Module 1: Book Data Pipeline

## 1. Project Overview

This module collects book information from Books to Scrape, cleans the collected data, stores it in a normalized SQLite database, and performs SQL and pandas analysis.

## 2. Data Collection

- Website: https://books.toscrape.com/
- Pages scraped: First 5 pages
- Total books collected: 100
- Fields collected:
  - Title
  - Price
  - Star rating
  - Availability
  - Category

## 3. Data Cleaning

The raw data was cleaned using pandas.

Cleaning operations:
- Converted book prices into numeric GBP values.
- Converted star ratings into integers from 1 to 5.
- Created a Boolean in_stock column.
- Converted GBP prices into INR using the fixed rate of 1 GBP = 105.50 INR.
- Checked for missing values.
- Removed rows with missing titles or categories.

Output file: books_clean.csv

## 4. Database Design

The cleaned data was stored in SQLite.

Two normalized tables were created:

### categories
- category_id: Primary key
- category_name: Unique category name

### books
- book_id: Primary key
- title
- price_gbp
- rating
- availability
- in_stock
- price_inr
- category_id: Foreign key referencing categories

Database file: books.db

Database records:
- Books: 100
- Categories: 29

## 5. SQL Queries

The following SQL operations were demonstrated:

1. SELECT and WHERE: Books costing more than £40.
2. ORDER BY: Books sorted by price.
3. LIMIT: First 10 books.
4. DISTINCT: Unique book categories.
5. IN: Books with ratings of 4 or 5.
6. JOIN: Books combined with their category names.

Query results are saved in sql_query_results.txt.

## 6. Pandas and SQL Comparison

Two SQL query results were loaded into pandas using pd.read_sql().

The SQL JOIN was reproduced using pandas merge().

The results were compared to verify that both approaches produced identical data.

## 7. How to Run

Run these commands from the project root folder:

```bash
python data_pipeline/scraper.py
python data_pipeline/clean_data.py
python data_pipeline/database.py
python data_pipeline/queries.py
python data_pipeline/pandas_sql.py
```

## 8. Files

- scraper.py: Collects book information.
- books_raw.csv: Raw scraped data.
- clean_data.py: Cleans the collected data.
- books_clean.csv: Cleaned data.
- database.py: Creates and populates the SQLite database.
- books.db: SQLite database.
- queries.py: Executes SQL queries.
- sql_query_results.txt: Saved query results.
- pandas_sql.py: Reads SQL data into pandas and compares JOIN results.
- README.md: Module documentation.
