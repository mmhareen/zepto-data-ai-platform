import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
response = requests.get(url)
response.encoding = "utf-8"

soup = BeautifulSoup(response.text, "html.parser")
books = soup.find_all("article", class_="product_pod")

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

book_data = []

for book in books:
    title = book.h3.a["title"]
    price_text = book.find("p", class_="price_color").text
    rating_word = book.find("p", class_="star-rating")["class"][1]
    availability_text = book.find("p", class_="instock").text.strip()

    price_gbp = float(price_text.replace("£", ""))
    rating = rating_map[rating_word]
    in_stock = "In stock" in availability_text

    book_data.append({
        "title": title,
        "price_gbp": price_gbp,
        "rating": rating,
        "in_stock": in_stock,
        "category": "Travel"
    })

for b in book_data:
    print(b)