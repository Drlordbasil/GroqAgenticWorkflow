import time
import random
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import trafilatura
from selenium.common.exceptions import WebDriverException, NoSuchElementException, TimeoutException
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging
import time
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

max_content_length = 5000  # Increased for more comprehensive results
max_retries = 3
retry_delay = 5
max_pages_per_site = 10  # Increased for more thorough crawling
max_search_results = 10  # Increased number of search results to process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SelectorRL:
    def __init__(self):
        self.selectors = {
            "google": {
                "search_box": ["input[name='q']", "textarea[name='q']", "#search-input"],
                "result": ["div.g", "div.tF2Cxc", "div.yuRUbf"]
            },
            "bing": {
                "search_box": ["input[name='q']", "#sb_form_q"],
                "result": ["li.b_algo", "div.b_title", "h2"]
            },
            "brave": {
                "search_box": ["input[name='q']", "#searchbox"],
                "result": ["div.snippet", "div.fdb", "div.result"]
            }
        }
        self.q_values = {engine: {selector: 0 for selector_type in selectors.values() for selector in selector_type} for engine, selectors in self.selectors.items()}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def get_selector(self, engine, selector_type):
        if random.random() < self.epsilon:
            return random.choice(self.selectors[engine][selector_type])
        else:
            return max(self.selectors[engine][selector_type], key=lambda s: self.q_values[engine][s])

    def update_q_value(self, engine, selector, reward):
        self.q_values[engine][selector] += self.learning_rate * (reward - self.q_values[engine][selector])

    def add_new_selector(self, engine, selector_type, new_selector):
        if new_selector not in self.selectors[engine][selector_type]:
            self.selectors[engine][selector_type].append(new_selector)
            self.q_values[engine][new_selector] = 0

    def save_state(self, filename='selector_rl_state.json'):
        state = {
            'q_values': self.q_values,
            'selectors': self.selectors
        }
        with open(filename, 'w') as f:
            json.dump(state, f)

    def load_state(self, filename='selector_rl_state.json'):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            self.q_values = state['q_values']
            self.selectors = state['selectors']
        except FileNotFoundError:
            logging.info("No saved state found. Starting with default values.")
class WebResearchTool:
    def __init__(self, max_content_length=max_content_length):
        self.max_content_length = max_content_length
        self.selector_rl = SelectorRL()
        self.selector_rl.load_state()
        self.vectorizer = TfidfVectorizer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _initialize_webdriver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-browser-side-navigation')
        options.add_argument('--disable-features=VizDisplayCompositor')
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def find_new_selector(self, driver, element_type):
        if element_type == "search_box":
            potential_selectors = driver.find_elements(By.XPATH, "//input[@type='text'] | //input[@type='search'] | //textarea")
        else:  # result
            potential_selectors = driver.find_elements(By.XPATH, "//div[.//a] | //li[.//a] | //h2[.//a]")

        for element in potential_selectors:
            try:
                selector = self.get_css_selector(driver, element)
                return selector
            except:
                continue
        return None

    def get_css_selector(self, driver, element):
        return driver.execute_script("""
            var path = [];
            var element = arguments[0];
            while (element.nodeType === Node.ELEMENT_NODE) {
                var selector = element.nodeName.toLowerCase();
                if (element.id) {
                    selector += '#' + element.id;
                    path.unshift(selector);
                    break;
                } else {
                    var sibling = element;
                    var nth = 1;
                    while (sibling.previousElementSibling) {
                        sibling = sibling.previousElementSibling;
                        if (sibling.nodeName.toLowerCase() == selector)
                            nth++;
                    }
                    if (nth != 1)
                        selector += ":nth-of-type("+nth+")";
                }
                path.unshift(selector);
                element = element.parentNode;
            }
            return path.join(' > ');
        """, element)

    def extract_text_from_url(self, url):
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
                if text and len(text) >= 50:
                    return text

                driver = self._initialize_webdriver()
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
                    element.decompose()
                text = ' '.join(p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']) if len(p.get_text().strip()) > 20)
                return text if len(text) >= 50 else None
            except RequestException as e:
                logging.warning(f"Error extracting text from URL {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to extract text from URL {url} after {max_retries} attempts")
                    return None
            finally:
                if 'driver' in locals():
                    driver.quit()

    def crawl_website(self, url, max_pages=max_pages_per_site):
        visited = set()
        to_visit = [url]
        graph = nx.DiGraph()
        content = {}

        try:
            while to_visit and len(visited) < max_pages:
                current_url = to_visit.pop(0)
                if current_url in visited:
                    continue

                visited.add(current_url)

                try:
                    response = self.session.get(current_url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract content
                    text = self.extract_text_from_url(current_url)
                    if text:
                        content[current_url] = text

                    # Find links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(current_url, href)
                        if full_url.startswith(url):  # Stay on the same domain
                            graph.add_edge(current_url, full_url)
                            if full_url not in visited:
                                to_visit.append(full_url)

                except RequestException as e:
                    logging.warning(f"Error crawling {current_url}: {e}")

                time.sleep(1)  # Add a delay to avoid overwhelming the server

            return graph, content
        except Exception as e:
            logging.error(f"Error in crawl_website: {e}")
            return nx.DiGraph(), {}

    def calculate_similarity(self, query, text):
        tfidf_matrix = self.vectorizer.fit_transform([query, text])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def process_search_result(self, result, engine_name, query):
        link = result.select_one('a')
        if link and link.get('href'):
            url = link['href']
            if url.startswith('http'):
                try:
                    graph, crawled_content = self.crawl_website(url)
                    search_results = []
                    for page_url, content in crawled_content.items():
                        similarity = self.calculate_similarity(query, content)
                        if similarity > 0.1:  # Adjust threshold as needed
                            search_results.append({
                                "title": link.get_text(),
                                "link": page_url,
                                "content": content,
                                "similarity": similarity
                            })
                    return search_results
                except Exception as e:
                    logging.error(f"Error processing search result from {engine_name}: {e}")
        return []

    def web_research(self, query):
        combined_query = query
        search_engines = [
            ("https://www.google.com/search", "google"),
            ("https://www.bing.com/search", "bing"),
            ("https://search.brave.com/search", "brave")
        ]
        all_search_results = []

        for engine_url, engine_name in search_engines:
            try:
                driver = self._initialize_webdriver()
                driver.get(engine_url)
                
                search_box_selector = self.selector_rl.get_selector(engine_name, "search_box")
                result_selector = self.selector_rl.get_selector(engine_name, "result")
                
                try:
                    search_box = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, search_box_selector))
                    )
                    search_box.send_keys(combined_query)
                    search_box.send_keys(Keys.RETURN)
                    
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, result_selector))
                    )
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    results = soup.select(result_selector)[:max_search_results]
                    
                    if results:
                        self.selector_rl.update_q_value(engine_name, search_box_selector, 1)
                        self.selector_rl.update_q_value(engine_name, result_selector, 1)
                    else:
                        new_search_box_selector = self.find_new_selector(driver, "search_box")
                        new_result_selector = self.find_new_selector(driver, "result")
                        
                        if new_search_box_selector and new_result_selector:
                            self.selector_rl.add_new_selector(engine_name, "search_box", new_search_box_selector)
                            self.selector_rl.add_new_selector(engine_name, "result", new_result_selector)

                            # Retry with new selectors
                            driver.get(engine_url)
                            search_box = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, new_search_box_selector))
                            )
                            search_box.send_keys(combined_query)
                            search_box.send_keys(Keys.RETURN)
                            
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, new_result_selector))
                            )
                            
                            soup = BeautifulSoup(driver.page_source, 'html.parser')
                            results = soup.select(new_result_selector)[:max_search_results]
                    
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_result = {executor.submit(self.process_search_result, result, engine_name, query): result for result in results}
                        for future in as_completed(future_to_result):
                            all_search_results.extend(future.result())

                except (NoSuchElementException, TimeoutException) as e:
                    logging.error(f"Error with {engine_name} search: {e}")
                    self.selector_rl.update_q_value(engine_name, search_box_selector, -1)
                    self.selector_rl.update_q_value(engine_name, result_selector, -1)
            except WebDriverException as e:
                logging.error(f"Error with {engine_name} search: {e}")
            finally:
                driver.quit()

            time.sleep(2)  # Add a delay between search engine queries

        self.selector_rl.save_state()

        if not all_search_results:
            return f"No results found for the query: {combined_query}"

        # Sort results by similarity
        all_search_results.sort(key=lambda x: x['similarity'], reverse=True)

        aggregated_content = ""
        for result in all_search_results:
            if len(aggregated_content) + len(result['content']) <= self.max_content_length:
                aggregated_content += f"[Source: {result['link']}]\n{result['content']}\n\n"
            else:
                remaining_chars = self.max_content_length - len(aggregated_content)
                aggregated_content += f"[Source: {result['link']}]\n{result['content'][:remaining_chars]}"
                break

        return self.summarize_results(aggregated_content, combined_query)

    def summarize_results(self, aggregated_content, query):
        # This method can be expanded to provide a more concise summary of the research results
        summary = f"Research results for query: {query}\n\n"
        summary += "Key findings:\n"
        
        # Extract main points (this is a simple implementation and can be improved)
        sentences = aggregated_content.split('.')
        main_points = [sentence.strip() for sentence in sentences if query.lower() in sentence.lower()][:5]
        
        for i, point in enumerate(main_points, 1):
            summary += f"{i}. {point}.\n"
        
        summary += f"\nDetailed information:\n{aggregated_content}"
        
        return summary

if __name__ == "__main__":
    tool = WebResearchTool()
    query = "How to create a website"
    print(tool.web_research(query))