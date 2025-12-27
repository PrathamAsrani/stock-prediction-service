import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class NewsCollector:
    """Collect news articles about stocks"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_google_news(self, symbol: str, max_results: int = 10) -> List[Dict]:
        """Fetch news from Google News"""
        try:
            query = f"{symbol} stock news"
            url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            for article in soup.find_all('article')[:max_results]:
                try:
                    title_elem = article.find('h3')
                    if title_elem:
                        articles.append({
                            'title': title_elem.get_text(),
                            'source': 'Google News',
                            'timestamp': datetime.now()
                        })
                except:
                    continue
            
            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def fetch_moneycontrol_news(self, symbol: str) -> List[Dict]:
        """Fetch news from MoneyControl (placeholder)"""
        # Implement MoneyControl scraping
        return []
    
    def fetch_economic_times_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Economic Times (placeholder)"""
        # Implement ET scraping
        return []
    