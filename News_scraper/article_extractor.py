from newspaper import Article, Config
from typing import Dict, Optional
import requests

class ArticleExtractor:
    def __init__(self):
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        self.config.request_timeout = 15
        self.config.fetch_images = False
        self.config.memoize_articles = False
    
    def extract(self, url: str) -> Optional[Dict]:
        """Extract article content using Newspaper4k."""
        # Skip problematic URLs
        if not url or 'google.com' in url or url.endswith('/'):
            return None
            
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            
            # Only run NLP if we have text
            if article.text and len(article.text) > 100:
                try:
                    article.nlp()
                except Exception:
                    pass  # NLP is optional, continue without it
            
            # Check if we got meaningful content
            if not article.text or len(article.text) < 50:
                print(f"   ⚠️ Article too short or empty: {url[:50]}")
                return None
            
            return {
                'url': url,
                'title': article.title or 'Unknown Title',
                'text': article.text,
                'summary': getattr(article, 'summary', '') or '',
                'keywords': getattr(article, 'keywords', []) or [],
                'authors': article.authors or [],
                'publish_date': str(article.publish_date) if article.publish_date else None
            }
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Network error for {url[:50]}: {type(e).__name__}")
            return None
        except Exception as e:
            print(f"   ⚠️ Extraction error for {url[:50]}: {type(e).__name__}")
            return None
    
    def extract_basic(self, url: str, headline: str = "") -> Optional[Dict]:
        """Fallback extraction - just use the headline if article fails."""
        return {
            'url': url,
            'title': headline,
            'text': headline,  # Use headline as minimal text for sentiment
            'summary': '',
            'keywords': [],
            'authors': [],
            'publish_date': None
        }
