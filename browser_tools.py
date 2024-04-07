from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup

class BrowserTools:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def search_google(self, query):
        search_url = f"https://www.google.com/search?q={query}"
        self.driver.get(search_url)
        search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.yuRUbf > a")
        top_urls = [result.get_attribute("href") for result in search_results[:3]]
        return top_urls

    def scrape_page(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        title = soup.title.string if soup.title else ""
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        content = " ".join(paragraphs)

        return {
            "url": url,
            "title": title,
            "content": content
        }

    def research_topic(self, topic):
        search_results = self.search_google(topic)
        scraped_pages = [self.scrape_page(url) for url in search_results]
        return scraped_pages

    def close(self):
        self.driver.quit()