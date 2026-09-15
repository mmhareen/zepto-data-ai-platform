import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_categories():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    category_tags = soup.select("div.side_categories ul li ul li a")

    categories = []
    for tag in category_tags:
        name = tag.text.strip()
        full_url = urljoin(BASE_URL, tag["href"])
        categories.append({"name": name, "url": full_url})
    return categories


def scrape_page(url, category_name):
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    page_data = []
    for book in books:
        title = book.h3.a["title"]
        price_text = book.find("p", class_="price_color").text
        rating_word = book.find("p", class_="star-rating")["class"][1]
        availability_text = book.find("p", class_="instock").text.strip()

        page_data.append({
            "title": title,
            "price_gbp": float(price_text.replace("£", "")),
            "rating": rating_map[rating_word],
            "in_stock": "In stock" in availability_text,
            "category": category_name
        })

    next_button = soup.find("li", class_="next")
    next_url = urljoin(url, next_button.a["href"]) if next_button else None

    return page_data, next_url


if __name__ == "__main__":
    categories = get_categories()
    all_books = []

    for cat in categories[:3]:
        url = cat["url"]
        while url:
            page_data, url = scrape_page(url, cat["name"])
            all_books.extend(page_data)

    print(f"Total books scraped: {len(all_books)}")

if __name__ == "__main__":
    categories = get_categories()
    all_books = []

    for cat in categories[:3]:
        url = cat["url"]
        while url:
            page_data, url = scrape_page(url, cat["name"])
            all_books.extend(page_data)

    print(f"Total books scraped: {len(all_books)}")

df = pd.DataFrame(all_books)
df["price_inr"] = df["price_gbp"] * 105.50
print(df.head())
print(df.dtypes)

df.to_csv("data_pipeline/books_raw.csv", index=False)
print("Saved to data_pipeline/books_raw.csv")