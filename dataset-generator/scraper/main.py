import requests
from bs4 import BeautifulSoup
from category import getCategoryLinks

def getMainPage(url):
    # Make the request
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all 'a' tags with href starting with '/list'
        list_links = soup.find_all('a', href=lambda href: href and href.startswith('/archive'))

        # Extract the href attributes
        hrefs = [link['href'] for link in list_links]

        # Print the hrefs
        for href in hrefs:
            fullhref = "https://arxiv.org" + href
            print(fullhref)
            getCategoryLinks(fullhref)
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

    

# Define the URL
url = "https://arxiv.org/"  # Replace this with the actual URL
getCategoryLinks(url)


