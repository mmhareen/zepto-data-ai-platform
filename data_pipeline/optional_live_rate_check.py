"""OPTIONAL, UNGRADED extension for Module 1.

Demonstrates explicit HTTP status-code handling with a free, keyless
currency API, falling back to the assignment's fixed rate (105.50) on
any failure. This script is entirely separate from scrape.py: it does
not modify books_raw.csv or price_inr in any way. The required
submission's price_inr column is, and remains, computed only from the
fixed 105.50 rate in scrape.py.
"""
import requests

FIXED_RATE = 105.50
LIVE_RATE_URL = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest"
    "/v1/currencies/gbp.json"
)
REQUEST_TIMEOUT = 10


def get_gbp_to_inr_rate():
    """Try to fetch today's live GBP->INR rate from a free, keyless API.

    Explicitly checks the HTTP status code before trusting the response.
    Returns (rate, source_label) where source_label is "live" or "fixed
    fallback" so the caller/printout is always honest about which was used.
    """
    try:
        response = requests.get(LIVE_RATE_URL, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        print(f"Live rate lookup failed (network error: {exc}). Falling back to fixed rate.")
        return FIXED_RATE, "fixed fallback"

    if response.status_code != 200:
        print(f"Live rate lookup failed (HTTP {response.status_code}). Falling back to fixed rate.")
        return FIXED_RATE, "fixed fallback"

    try:
        data = response.json()
        live_rate = data["gbp"]["inr"]
    except (KeyError, ValueError) as exc:
        print(f"Live rate lookup failed (unexpected response shape: {exc}). Falling back to fixed rate.")
        return FIXED_RATE, "fixed fallback"

    return live_rate, "live"


def main():
    rate, source = get_gbp_to_inr_rate()
    print(f"GBP -> INR rate used: {rate} (source: {source})")
    print(f"Assignment's required fixed rate for grading: {FIXED_RATE}")
    if source == "live":
        difference = rate - FIXED_RATE
        print(f"Difference from fixed rate: {difference:+.4f}")
    print("\nNote: this script is purely a demonstration of the optional "
          "extension. It does not affect books_raw.csv or the graded "
          "price_inr column, which is computed only from the fixed rate "
          "in scrape.py.")


if __name__ == "__main__":
    main()
