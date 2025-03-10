# Dataset Generator and Scraper

This directory contains scripts for scraping research papers from arXiv and downloading them as PDFs.

## Files and Their Purpose

### `main.py`
- **Purpose**: Entry point for scraping the main page of arXiv to get category links.
- **Functionality**: 
  - Fetches the main page.
  - Extracts and prints category links.
  - Calls `getCategoryLinks` to process each category link.

### `category.py`
- **Purpose**: Handles scraping of category pages to get paper links.
- **Functionality**: 
  - Fetches category pages.
  - Extracts and prints paper links.
  - Calls `getAllPapers` to download papers from each link.

### `papers.py`
- **Purpose**: Handles downloading of individual research papers.
- **Functionality**: 
  - Fetches paper links.
  - Downloads and saves the papers as PDFs.

## Usage

1. **Scraping Papers**:
   - Run `main.py` to start the scraping process.
   - The script will navigate through category pages and download the first two papers from each category.

## Requirements

- Python 3.x
- `requests` library
- `beautifulsoup4` library

Install the required libraries using:
```sh
pip install requests beautifulsoup4
```

## Notes

- Ensure you have an internet connection to scrape and download papers.
- ⚠️ WARNING: Be careful with using the scraper without a stop mechanism, as it will download every paper it can find (Possible memory overflow)
