import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time

# Website address
BASE_URL = "https://books.toscrape.com/"

# Browser information
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Store collected books
all_books = []

# Collect the first five pages
for page_number in range(1, 6):

    page_url = urljoin(
        BASE_URL,
        f"catalogue/page-{page_number}.html"
    )

    print(f"Collecting page {page_number}...")

    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Find book cards
    book_cards = soup.select("article.product_pod")

    for card in book_cards:

        # Book title
        title = card.h3.a["title"]

        # Book price
        price = card.select_one(
            "p.price_color"
        ).get_text(strip=True)

        # Star rating
        rating_classes = card.select_one(
            "p.star-rating"
        )["class"]

        star_rating = next(
            (word for word in rating_classes if word != "star-rating"),
            None
        )

        # Book detail page
        book_link = card.h3.a["href"]

        book_url = urljoin(page_url, book_link)

        book_response = requests.get(
            book_url,
            headers=HEADERS,
            timeout=20
        )

        book_response.raise_for_status()

        book_soup = BeautifulSoup(
            book_response.text,
            "html.parser"
        )

        # Book category
        breadcrumb = book_soup.select(
            "ul.breadcrumb li a"
        )

        category = (
            breadcrumb[-1].get_text(strip=True)
            if len(breadcrumb) >= 3
            else "Unknown"
        )

        # Availability
        availability_element = book_soup.select_one(
            "p.instock.availability"
        )

        availability = (
            availability_element.get_text(" ", strip=True)
            if availability_element
            else "Unknown"
        )

        # Save information
        all_books.append({
            "title": title,
            "price_gbp": price,
            "star_rating": star_rating,
            "availability": availability,
            "category": category
        })

        time.sleep(0.1)

# Convert records into a DataFrame
books_df = pd.DataFrame(all_books)

# Save CSV file
books_df.to_csv(
    "data_pipeline/books_raw.csv",
    index=False
)

# Display results
print("\nScraping completed!")
print("Total books collected:", len(books_df))

print("\nFirst five books:")
print(books_df.head())

print("\nCSV file saved successfully!")
