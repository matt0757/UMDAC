from newspaper import Article, Config
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup
import re
import time

class ArticleExtractor:
    def __init__(self):
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        self.config.request_timeout = 20
        self.config.fetch_images = False
        self.config.memoize_articles = False
        
        # Headers for direct requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        # Sites that need special handling
        self.problematic_domains = [
            'bloomberg.com',  # Paywall
            'wsj.com',        # Paywall
            'ft.com',         # Paywall
            'economist.com',  # Paywall
            'nytimes.com',    # Paywall
            'washingtonpost.com',  # Paywall
        ]
    
    def extract(self, url: str) -> Optional[Dict]:
        """Extract article content using multiple strategies."""
        # Skip problematic URLs
        if not url or 'google.com' in url or url.endswith('/'):
            return None
        
        # Check for paywall sites - try but expect limited content
        is_paywall_site = any(domain in url for domain in self.problematic_domains)
        if is_paywall_site:
            print(f"   ⚠️ Paywall site detected: {url[:50]}")
        
        # Strategy 1: Try newspaper4k first
        result = self._extract_newspaper(url)
        if result and len(result.get('text', '')) > 200:
            return result
        
        # Strategy 2: Try direct requests + BeautifulSoup
        result = self._extract_beautifulsoup(url)
        if result and len(result.get('text', '')) > 200:
            return result
        
        # Strategy 3: Try with different headers (mobile user agent)
        result = self._extract_mobile(url)
        if result and len(result.get('text', '')) > 200:
            return result
        
        # If we got some content but it's short, still return it
        if result and len(result.get('text', '')) > 50:
            return result
        
        return None
    
    def _extract_newspaper(self, url: str) -> Optional[Dict]:
        """Extract using newspaper4k library."""
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            
            # Only run NLP if we have text
            if article.text and len(article.text) > 100:
                try:
                    article.nlp()
                except Exception:
                    pass
            
            if not article.text or len(article.text) < 20:
                return None
            
            text_preview = article.text[:100].replace('\n', ' ')
            print(f"   ✓ Extracted {len(article.text)} chars: \"{text_preview}...\"")
            
            return {
                'url': url,
                'title': article.title or 'Unknown Title',
                'text': article.text,
                'summary': getattr(article, 'summary', '') or '',
                'keywords': getattr(article, 'keywords', []) or [],
                'authors': article.authors or [],
                'publish_date': str(article.publish_date) if article.publish_date else None
            }
            
        except Exception as e:
            return None
    
    def _extract_beautifulsoup(self, url: str) -> Optional[Dict]:
        """Extract using requests + BeautifulSoup as fallback."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                                          'aside', 'advertisement', 'iframe', 'noscript']):
                element.decompose()
            
            # Try to find the main article content
            article_text = ""
            title = ""
            
            # Get title
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Look for common article containers
            article_selectors = [
                'article',
                '[role="main"]',
                '.article-body',
                '.article-content',
                '.story-body',
                '.post-content',
                '.entry-content',
                '.content-body',
                '#article-body',
                '.article__body',
                '.story-content',
                'main',
            ]
            
            for selector in article_selectors:
                container = soup.select_one(selector)
                if container:
                    # Get all paragraph text
                    paragraphs = container.find_all('p')
                    text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                    if len(text) > len(article_text):
                        article_text = text
            
            # Fallback: get all paragraphs if no article container found
            if len(article_text) < 100:
                paragraphs = soup.find_all('p')
                # Filter out short paragraphs (likely navigation, etc.)
                good_paragraphs = [p.get_text().strip() for p in paragraphs 
                                  if len(p.get_text().strip()) > 50]
                article_text = ' '.join(good_paragraphs)
            
            # Clean up the text
            article_text = self._clean_text(article_text)
            
            if len(article_text) < 50:
                return None
            
            text_preview = article_text[:100].replace('\n', ' ')
            print(f"   ✓ [BS4] Extracted {len(article_text)} chars: \"{text_preview}...\"")
            
            return {
                'url': url,
                'title': title or 'Unknown Title',
                'text': article_text,
                'summary': '',
                'keywords': [],
                'authors': [],
                'publish_date': None
            }
            
        except Exception as e:
            return None
    
    def _extract_mobile(self, url: str) -> Optional[Dict]:
        """Try extraction with mobile user agent (sometimes bypasses restrictions)."""
        mobile_headers = self.headers.copy()
        mobile_headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
        
        try:
            response = requests.get(url, headers=mobile_headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get title
            title = ""
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Get paragraphs
            paragraphs = soup.find_all('p')
            good_paragraphs = [p.get_text().strip() for p in paragraphs 
                              if len(p.get_text().strip()) > 40]
            article_text = ' '.join(good_paragraphs)
            article_text = self._clean_text(article_text)
            
            if len(article_text) < 50:
                return None
            
            text_preview = article_text[:100].replace('\n', ' ')
            print(f"   ✓ [Mobile] Extracted {len(article_text)} chars: \"{text_preview}...\"")
            
            return {
                'url': url,
                'title': title or 'Unknown Title',
                'text': article_text,
                'summary': '',
                'keywords': [],
                'authors': [],
                'publish_date': None
            }
            
        except Exception as e:
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common boilerplate phrases
        boilerplate = [
            'cookie', 'privacy policy', 'terms of service', 'subscribe',
            'sign up', 'newsletter', 'advertisement', 'sponsored',
            'read more', 'continue reading', 'click here'
        ]
        lines = text.split('. ')
        cleaned_lines = [line for line in lines 
                        if not any(bp in line.lower() for bp in boilerplate)]
        text = '. '.join(cleaned_lines)
        return text.strip()
    
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

