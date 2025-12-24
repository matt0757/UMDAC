import asyncio
from news_scraper import NewsScraper
from article_extractor import ArticleExtractor
from sentiment_analyzer import SentimentAnalyzer
from typing import List, Optional
import json

class EconomicNewsSentimentAnalyzer:
    def __init__(self, fast_mode: bool = True, headline_threshold: float = 0.5):
        """
        Initialize the analyzer.
        
        Args:
            fast_mode: If True, headlines with strong sentiment (above threshold) 
                      skip full article extraction for faster processing.
            headline_threshold: Minimum sentiment magnitude (0-1) to accept a 
                              headline without full article extraction.
        """
        self.scraper = NewsScraper()
        self.extractor = ArticleExtractor()
        self.analyzer = SentimentAnalyzer()
        self.fast_mode = fast_mode
        self.headline_threshold = headline_threshold
        
    async def analyze_economic_news(
        self, 
        keywords: List[str] = None, 
        max_articles: int = 10,
        fast_mode: bool = None
    ) -> dict:
        """Main method to scrape and analyze economic news.
        
        Args:
            keywords: Search keywords
            max_articles: Target number of articles
            fast_mode: Override instance fast_mode setting
        """
        
        if keywords is None:
            keywords = ["US"]
        
        use_fast_mode = fast_mode if fast_mode is not None else self.fast_mode
        
        if use_fast_mode:
            print(f"🔍 Searching for news with keywords: {keywords} [FAST MODE - threshold: {self.headline_threshold}]")
        else:
            print(f"🔍 Searching for news with keywords: {keywords}")
        
        # Step 1: Scrape headlines (get extra in case some fail)
        buffer_multiplier = 2 if use_fast_mode else 3
        headlines = await self.scraper.search_headlines(keywords, (max_articles + 2) * buffer_multiplier)
        print(f"📰 Found {len(headlines)} headlines")
        
        # Step 2: Process headlines - fast mode pre-analyzes headlines first
        results = []  # Strong results that count towards quota
        backup_results = []  # Weak headline-only results stored as backup
        skipped = 0
        full_extractions = 0
        fast_accepts = 0
        
        for i, headline in enumerate(headlines):
            # Stop if we have enough articles
            if len(results) >= max_articles:
                break
            
            headline_text = headline['headline']
            print(f"📄 Processing ({len(results)}/{max_articles}, attempt {i+1}): {headline_text[:50]}...")
            
            try:
                headline_sentiment = None
                headline_score = 0
                
                # FAST MODE: Pre-analyze headline sentiment first
                if use_fast_mode:
                    headline_sentiment = self._quick_headline_analysis(headline_text)
                    headline_score = abs(headline_sentiment.get('final_score', 0))
                    
                    # If headline has strong sentiment, accept it without full extraction
                    if headline_score >= self.headline_threshold:
                        headline_sentiment['url'] = headline.get('url', '')
                        headline_sentiment['source'] = headline.get('source', 'unknown')
                        headline_sentiment['analysis_type'] = 'headline_fast'
                        headline_sentiment['text_length'] = len(headline_text)
                        results.append(headline_sentiment)
                        fast_accepts += 1
                        print(f"   ⚡ FAST: {headline_sentiment['category']} (score: {headline_sentiment['final_score']}) [Strong headline - skipped extraction]")
                        continue
                    else:
                        print(f"   🔍 Weak headline ({headline_score:.2f} < {self.headline_threshold}), trying full extraction...")
                
                # Get actual URL (follow redirect)
                actual_url = await self.scraper.get_article_url(headline['url'])
                
                # Skip invalid URLs
                if not actual_url or 'chrome-error' in actual_url or (actual_url == headline['url'] and 'news.google.com' in headline['url']):
                    # Store weak headline as backup, don't count towards quota
                    if use_fast_mode and headline_sentiment:
                        headline_sentiment['url'] = headline.get('url', '')
                        headline_sentiment['source'] = headline.get('source', 'unknown')
                        headline_sentiment['analysis_type'] = 'headline_backup'
                        backup_results.append(headline_sentiment)
                        print(f"   ⚠️ URL failed, stored as backup (score: {headline_sentiment['final_score']})")
                    else:
                        print(f"   ⚠️ Invalid URL, skipping...")
                        skipped += 1
                    continue
                
                # Extract article content
                article = self.extractor.extract(actual_url)
                
                if article and article.get('text') and len(article.get('text', '')) > 200:
                    # Analyze full article sentiment - counts towards quota
                    sentiment = self.analyzer.analyze(article)
                    sentiment['source'] = headline.get('source', 'unknown')
                    sentiment['analysis_type'] = 'full_article'
                    results.append(sentiment)
                    full_extractions += 1
                    print(f"   → {sentiment['category']} (score: {sentiment['final_score']}) [Full: {sentiment.get('text_length', 0)} chars] ✓")
                else:
                    # Extraction failed - store as backup, don't count towards quota
                    if use_fast_mode and headline_sentiment:
                        headline_sentiment['url'] = actual_url or headline.get('url', '')
                        headline_sentiment['source'] = headline.get('source', 'unknown')
                        headline_sentiment['analysis_type'] = 'headline_backup'
                        backup_results.append(headline_sentiment)
                        print(f"   ℹ️ Extraction failed, stored as backup (score: {headline_sentiment['final_score']})")
                    else:
                        # Standard mode - analyze headline now and store as backup
                        article = self.extractor.extract_basic(actual_url, headline_text)
                        if article:
                            sentiment = self.analyzer.analyze(article)
                            sentiment['source'] = headline.get('source', 'unknown')
                            sentiment['analysis_type'] = 'headline_backup'
                            backup_results.append(sentiment)
                            print(f"   ℹ️ Extraction failed, stored as backup (score: {sentiment['final_score']})")
                        else:
                            skipped += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error: {e} - skipping...")
                skipped += 1
        
        # Step 3: Fill remaining slots with backups if needed
        remaining_slots = max_articles - len(results)
        if remaining_slots > 0 and backup_results:
            # Sort backups by absolute score (strongest sentiment first)
            backup_results.sort(key=lambda x: abs(x.get('final_score', 0)), reverse=True)
            backups_used = backup_results[:remaining_slots]
            for backup in backups_used:
                backup['analysis_type'] = 'headline_only'  # Rename for final output
            results.extend(backups_used)
            print(f"📝 Added {len(backups_used)} backup headlines to fill remaining slots")
        
        # Summary
        headline_only_count = len([r for r in results if r.get('analysis_type') in ['headline_fast', 'headline_only']])
        if use_fast_mode:
            print(f"✅ Processed {len(results)} articles: {fast_accepts} fast (strong headline), {full_extractions} full extraction, {len(results) - fast_accepts - full_extractions} from backup, {skipped} skipped")
        else:
            print(f"✅ Processed {len(results)} articles: {full_extractions} full, {headline_only_count} headline-only, {skipped} skipped")
        
        # Step 4: Generate summary
        summary = self.analyzer.summarize_results(results)
        
        return {
            'summary': summary,
            'articles': results
        }
    
    def _quick_headline_analysis(self, headline_text: str) -> dict:
        """Quickly analyze a headline without full article extraction.
        
        Returns a sentiment dict that can be used directly if score is strong enough.
        """
        # Create a minimal article dict with just the headline
        article = {
            'title': headline_text,
            'text': headline_text,
            'url': '',
            'publish_date': None
        }
        
        # Use the same analyzer
        sentiment = self.analyzer.analyze(article)
        return sentiment
    
    def print_report(self, analysis: dict):
        """Print a formatted report."""
        print("\n" + "="*60)
        print("📊 ECONOMIC SENTIMENT ANALYSIS REPORT")
        print("="*60)
        
        summary = analysis['summary']
        
        # Handle empty results
        if summary.get('overall') == 'NO DATA' or not analysis.get('articles'):
            print("\n⚠️ No articles were successfully analyzed.")
            print("="*60 + "\n")
            return
        
        print(f"\n🎯 OVERALL: {summary['overall_assessment']}")
        print(f"   Average Score: {summary['average_score']} (-1 to 1 scale)")
        print(f"   Articles Analyzed: {summary['total_articles']}")
        print(f"   Breakdown: {summary['sentiment_ratio']}")
        
        print("\n" + "-"*60)
        print("📰 INDIVIDUAL ARTICLES:")
        print("-"*60)
        
        for article in analysis['articles']:
            emoji = "🟢" if article['category'] == "POSITIVE" else "🔴" if article['category'] == "NEGATIVE" else "🟡"
            title = article['title'][:70] if len(article['title']) > 70 else article['title']
            print(f"\n{emoji} {title}...")
            
            # Display publish date if available
            pub_date = article.get('publish_date')
            if pub_date and pub_date != 'None':
                print(f"   📅 Date: {pub_date}")
            else:
                print(f"   📅 Date: Not available")
            
            print(f"   Score: {article['final_score']} | {article['impact']}")
            print(f"   Confidence: {article.get('confidence', 'N/A')}")
            print(f"   Source: {article.get('source', 'unknown')}")
            
            # Show analysis type
            analysis_type = article.get('analysis_type', 'unknown')
            if analysis_type == 'full_article':
                print(f"   📝 Analysis: Full article ({article.get('text_length', 0)} chars)")
            else:
                print(f"   📝 Analysis: Headline only")
            
            # Display URL
            url = article.get('url', '')
            if url:
                print(f"   🔗 URL: {url}")
        
        print("\n" + "="*60)
        
        # Final verdict
        score = summary['average_score']
        if score > 0.2:
            verdict = "POSITIVE"
        elif score < -0.2:
            verdict = "NEGATIVE"
        else:
            verdict = "UNCERTAIN"
        
        print(verdict)
        print("="*60 + "\n")


# ============================================================
# PUBLIC API - Use these functions from your main app
# ============================================================

def analyze_news_sync(
    keywords: List[str] = None, 
    max_articles: int = 10,
    print_report: bool = False,
    save_json: bool = False,
    json_path: str = 'sentiment_report.json',
    fast_mode: bool = True,
    headline_threshold: float = 0.5
) -> dict:
    """
    Synchronous wrapper for news sentiment analysis.
    Use this from a regular (non-async) main function.
    
    Args:
        keywords: List of search terms (default: ["US economy"])
        max_articles: Maximum articles to analyze
        print_report: Whether to print formatted report
        save_json: Whether to save results to JSON file
        json_path: Path for JSON output
        fast_mode: If True, headlines with strong sentiment skip full extraction
        headline_threshold: Minimum sentiment magnitude (0-1) to skip extraction
    
    Returns:
        dict with 'summary' and 'articles' keys
        
    Example:
        from main_scraper import analyze_news_sync
        
        # Fast mode (default) - much faster
        results = analyze_news_sync(
            keywords=["US economy"],
            max_articles=10,
            fast_mode=True  # Headlines with score >= 0.5 skip extraction
        )
        
        # Thorough mode - always extract full articles
        results = analyze_news_sync(
            keywords=["US economy"],
            max_articles=10,
            fast_mode=False
        )
    """
    return asyncio.run(_analyze_news_async(
        keywords, max_articles, print_report, save_json, json_path,
        fast_mode, headline_threshold
    ))


async def analyze_news_async(
    keywords: List[str] = None, 
    max_articles: int = 10,
    print_report: bool = False,
    save_json: bool = False,
    json_path: str = 'sentiment_report.json',
    fast_mode: bool = True,
    headline_threshold: float = 0.5
) -> dict:
    """
    Async version for news sentiment analysis.
    Use this if your main app is already async.
    
    Args:
        keywords: List of search terms
        max_articles: Maximum articles to analyze
        print_report: Whether to print formatted report
        save_json: Whether to save results to JSON file
        json_path: Path for JSON output
        fast_mode: If True, headlines with strong sentiment skip full extraction
        headline_threshold: Minimum sentiment magnitude (0-1) to skip extraction
    
    Example:
        from main_scraper import analyze_news_async
        
        async def my_main():
            results = await analyze_news_async(keywords=["US economy"])
            print(results['summary']['overall_assessment'])
    """
    return await _analyze_news_async(
        keywords, max_articles, print_report, save_json, json_path,
        fast_mode, headline_threshold
    )


def get_sentiment_verdict(results: dict) -> str:
    """
    Get a simple verdict string from analysis results.
    
    Returns: "POSITIVE", "NEGATIVE", or "UNCERTAIN"
    """
    if not results or not results.get('summary'):
        return "UNCERTAIN"
    
    score = results['summary'].get('average_score', 0)
    if score > 0.2:
        return "POSITIVE"
    elif score < -0.2:
        return "NEGATIVE"
    else:
        return "UNCERTAIN"


def get_sentiment_score(results: dict) -> float:
    """
    Get the average sentiment score from analysis results.
    
    Returns: float between -1 and 1
    """
    if not results or not results.get('summary'):
        return 0.0
    return results['summary'].get('average_score', 0.0)


def get_article_urls(results: dict) -> List[str]:
    """
    Get list of all article URLs from analysis results.
    
    Returns: List of URL strings
    """
    if not results or not results.get('articles'):
        return []
    return [article.get('url', '') for article in results['articles'] if article.get('url')]


def get_articles_with_urls(results: dict) -> List[dict]:
    """
    Get list of articles with their titles, URLs, and sentiments.
    
    Returns: List of dicts with 'title', 'url', 'category', 'score' keys
    """
    if not results or not results.get('articles'):
        return []
    
    return [
        {
            'title': article.get('title', 'Unknown'),
            'url': article.get('url', ''),
            'category': article.get('category', 'NEUTRAL'),
            'score': article.get('final_score', 0),
            'source': article.get('source', 'unknown'),
            'publish_date': article.get('publish_date', None)
        }
        for article in results['articles']
    ]


def get_full_summary(results: dict) -> dict:
    """
    Get a complete summary including verdict, score, and all article URLs.
    
    Returns: dict with 'verdict', 'score', 'total_articles', 'articles' keys
    """
    return {
        'verdict': get_sentiment_verdict(results),
        'score': get_sentiment_score(results),
        'total_articles': len(results.get('articles', [])),
        'articles': get_articles_with_urls(results)
    }


# ============================================================
# INTERNAL HELPERS
# ============================================================

async def _analyze_news_async(
    keywords: List[str] = None, 
    max_articles: int = 10,
    print_report: bool = False,
    save_json: bool = False,
    json_path: str = 'sentiment_report.json',
    fast_mode: bool = True,
    headline_threshold: float = 0.5
) -> dict:
    """Internal async implementation."""
    if keywords is None:
        keywords = ["US economy"]
    
    analyzer = EconomicNewsSentimentAnalyzer(
        fast_mode=fast_mode,
        headline_threshold=headline_threshold
    )
    results = await analyzer.analyze_economic_news(
        keywords=keywords, 
        max_articles=max_articles
    )
    
    if print_report:
        analyzer.print_report(results)
    
    if save_json:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Full report saved to {json_path}")
    
    return results


# ============================================================
# STANDALONE EXECUTION
# ============================================================

async def main():
    """Run when executed directly."""
    results = await analyze_news_async(
        keywords=["US economy"],
        max_articles=3,
        print_report=True,
        save_json=False
    )


if __name__ == "__main__":
    import time
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"⏱️ Execution time: {end_time - start_time:.2f} seconds")
