from playwright.async_api import async_playwright
import asyncio
from typing import List, Dict
import urllib.parse
import random
import feedparser  # RSS feed parser

class NewsScraper:
    def __init__(self):
        self.news_sources = {
            "google_news": "https://news.google.com/search?q=",
            "bbc": "https://www.bbc.co.uk/search?q=",
            "ap_news": "https://apnews.com/search?q=",
        }
        # RSS feeds don't have captchas!
        self.rss_feeds = {
            "google_news_rss": "https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en",
            "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
            "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
        }
    
    async def search_headlines(self, keywords: List[str], max_results: int = 10) -> List[Dict]:
        """Search for headlines - tries RSS first, then web scraping as fallback."""
        articles = []
        
        # Try RSS feeds first (no captcha!)
        print("📡 Trying RSS feeds (captcha-free)...")
        for keyword in keywords:
            rss_articles = await self._scrape_rss_feeds(keyword, max_results)
            articles.extend(rss_articles)
        
        # If RSS didn't get enough results, try web scraping with stealth
        if len(articles) < max_results:
            print("🌐 Trying web scraping with stealth mode...")
            web_articles = await self._scrape_web_stealth(keywords, max_results - len(articles))
            articles.extend(web_articles)
        
        # Remove duplicates based on headline
        seen = set()
        unique_articles = []
        for article in articles:
            headline_lower = article['headline'].lower()
            if headline_lower not in seen:
                seen.add(headline_lower)
                unique_articles.append(article)
        
        return unique_articles[:max_results]
    
    async def _scrape_rss_feeds(self, keyword: str, max_results: int) -> List[Dict]:
        """Scrape headlines from RSS feeds - no captcha issues!"""
        articles = []
        keyword_lower = keyword.lower()
        
        # Google News RSS (keyword specific)
        try:
            google_rss_url = self.rss_feeds['google_news_rss'].format(keyword=urllib.parse.quote(keyword))
            feed = feedparser.parse(google_rss_url)
            
            for entry in feed.entries[:max_results]:
                articles.append({
                    'headline': entry.title,
                    'url': entry.link,
                    'keyword': keyword,
                    'source': 'google_news_rss',
                    'published': entry.get('published', None)
                })
            print(f"   [Google RSS] Found {len(feed.entries[:max_results])} headlines for '{keyword}'")
        except Exception as e:
            print(f"   [Google RSS] Error: {e}")
        
        # General business/economy RSS feeds
        general_feeds = ['bbc_business', 'cnbc', 'marketwatch']
        for feed_name in general_feeds:
            try:
                feed = feedparser.parse(self.rss_feeds[feed_name])
                count = 0
                for entry in feed.entries:
                    # Filter by keyword
                    title_lower = entry.title.lower()
                    if keyword_lower in title_lower or 'economy' in title_lower or 'economic' in title_lower:
                        articles.append({
                            'headline': entry.title,
                            'url': entry.link,
                            'keyword': keyword,
                            'source': feed_name,
                            'published': entry.get('published', None)
                        })
                        count += 1
                        if count >= 5:  # Limit per feed
                            break
                if count > 0:
                    print(f"   [{feed_name}] Found {count} relevant headlines")
            except Exception as e:
                print(f"   [{feed_name}] Error: {e}")
        
        return articles
    
    async def _scrape_web_stealth(self, keywords: List[str], max_results: int) -> List[Dict]:
        """Web scraping with stealth techniques."""
        articles = []
        
        async with async_playwright() as p:
            # Launch with stealth settings
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                ]
            )
            
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                permissions=['geolocation']
            )
            
            # Add stealth scripts
            await context.add_init_script("""
                // Overwrite navigator.webdriver
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Overwrite plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Overwrite languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
            """)
            
            page = await context.new_page()
            
            for keyword in keywords:
                # Add random delay to seem more human
                await page.wait_for_timeout(random.randint(2000, 4000))
                
                # Try AP News (usually less strict)
                ap_articles = await self._scrape_ap_news(page, keyword, max_results)
                articles.extend(ap_articles)
                    
            await browser.close()
        
        return articles
    
    async def _scrape_ap_news(self, page, keyword: str, max_results: int) -> List[Dict]:
        """Scrape headlines from AP News."""
        articles = []
        encoded_keyword = urllib.parse.quote(keyword)
        search_url = f"{self.news_sources['ap_news']}{encoded_keyword}"
        
        try:
            print(f"   [AP News] Navigating to: {search_url}")
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(random.randint(3000, 5000))
            
            # Random mouse movements to seem human
            await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
            await page.wait_for_timeout(random.randint(500, 1000))
            
            headlines = await page.evaluate("""
                () => {
                    const results = [];
                    
                    document.querySelectorAll('a[href*="/article/"]').forEach(link => {
                        const text = link.innerText.trim();
                        const href = link.getAttribute('href');
                        
                        if (text && text.length > 20 && text.length < 250 && href) {
                            const fullUrl = href.startsWith('http') ? href : 'https://apnews.com' + href;
                            if (!results.find(r => r.headline === text) && !results.find(r => r.url === fullUrl)) {
                                results.push({ headline: text, url: fullUrl });
                            }
                        }
                    });
                    
                    return results.slice(0, 15);
                }
            """)
            
            print(f"   [AP News] Found {len(headlines)} headlines for '{keyword}'")
            
            for item in headlines[:max_results]:
                item['keyword'] = keyword
                item['source'] = 'ap_news'
                articles.append(item)
                
        except Exception as e:
            print(f"   [AP News] Error: {e}")
        
        return articles
    
    async def get_article_url(self, url: str) -> str:
        """Follow redirects to get actual article URL."""
        # Most RSS URLs are direct
        if any(x in url for x in ['apnews.com', 'bbc.com', 'bbc.co.uk', 'cnbc.com', 'marketwatch.com', 'yahoo.com']):
            return url
        
        # Google News RSS URLs need redirect following
        if 'news.google.com' in url:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    await page.wait_for_timeout(2000)
                    final_url = page.url
                    
                    # Check if redirect failed
                    if 'chrome-error' in final_url or final_url == url or 'consent.google' in final_url:
                        print(f"   ⚠️ Redirect failed for: {url[:50]}...")
                        return ""  # Return empty to signal failure
                        
                except Exception as e:
                    print(f"   Could not follow redirect: {e}")
                    return ""  # Return empty to signal failure
                finally:
                    await browser.close()
                    
            return final_url
        
        return url
