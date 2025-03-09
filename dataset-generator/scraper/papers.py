import requests
from bs4 import BeautifulSoup
import os

def getAllPapers(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        list_links = soup.find_all('a', href=lambda href: href and href.startswith('/pdf'))
        
        # Extract the first two hrefs only
        hrefs = [link['href'] for link in list_links[:2]]

        # Download the PDFs
        for href in hrefs:
            full_url = "https://arxiv.org" + href
            print(f"Downloading: {full_url}")
            download_pdf(full_url)
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

def download_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_name = pdf_url.split("/")[-1] + ".pdf"
        pdf_path = os.path.join("allPapers", pdf_name)
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded: {pdf_name}")
    else:
        print(f"Failed to download {pdf_url}")
