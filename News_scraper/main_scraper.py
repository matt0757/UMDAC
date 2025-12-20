import asyncio
from news_scraper import NewsScraper
from article_extractor import ArticleExtractor
from sentiment_analyzer import SentimentAnalyzer
from typing import List, Optional
import json

class EconomicNewsSentimentAnalyzer:
    def __init__(self):
        self.scraper = NewsScraper()
        self.extractor = ArticleExtractor()
        self.analyzer = SentimentAnalyzer()
        
    async def analyze_economic_news(
        self, 
        keywords: List[str] = None, 
        max_articles: int = 10
    ) -> dict:
        """Main method to scrape and analyze economic news."""
        
        if keywords is None:
            keywords = ["US"]
        
        print(f"🔍 Searching for news with keywords: {keywords}")
        
        # Step 1: Scrape headlines (get extra in case some fail)
        buffer_multiplier = 3  # Get 3x more headlines as buffer for full article extraction
        headlines = await self.scraper.search_headlines(keywords, max_articles * buffer_multiplier)
        print(f"📰 Found {len(headlines)} headlines")
        
        # Step 2: Extract and analyze each article - prioritize full articles
        full_article_results = []
        headline_only_results = []
        skipped = 0
        
        for i, headline in enumerate(headlines):
            # Stop if we have enough full articles
            if len(full_article_results) >= max_articles:
                break
                
            current_full = len(full_article_results)
            current_total = current_full + len(headline_only_results)
            print(f"📄 Processing (full: {current_full}/{max_articles}, attempt {i+1}): {headline['headline'][:50]}...")
            
            try:
                # Get actual URL (follow redirect)
                actual_url = await self.scraper.get_article_url(headline['url'])
                
                # Skip invalid URLs
                if not actual_url or 'chrome-error' in actual_url or actual_url == headline['url'] and 'news.google.com' in headline['url']:
                    print(f"   ⚠️ Invalid URL, skipping to next article...")
                    skipped += 1
                    continue
                
                # Extract article content
                article = self.extractor.extract(actual_url)
                
                if article and article.get('text'):
                    # Analyze sentiment
                    sentiment = self.analyzer.analyze(article)
                    sentiment['source'] = headline.get('source', 'unknown')
                    
                    analysis_type = sentiment.get('analysis_type', 'unknown')
                    text_len = sentiment.get('text_length', 0)
                    
                    if analysis_type == 'full_article':
                        full_article_results.append(sentiment)
                        print(f"   → {sentiment['category']} (score: {sentiment['final_score']}) [Full article: {text_len} chars] ✓")
                    else:
                        # Store headline-only as backup
                        headline_only_results.append(sentiment)
                        print(f"   → {sentiment['category']} (score: {sentiment['final_score']}) [Headline only - stored as backup]")
                else:
                    # Try headline fallback but store separately
                    print(f"   ℹ️ Extraction failed, storing headline as backup")
                    article = self.extractor.extract_basic(actual_url, headline['headline'])
                    if article and article.get('text'):
                        sentiment = self.analyzer.analyze(article)
                        sentiment['source'] = headline.get('source', 'unknown')
                        headline_only_results.append(sentiment)
                        print(f"   → {sentiment['category']} (score: {sentiment['final_score']}) [Headline only - backup]")
                    else:
                        skipped += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error: {e} - skipping to next article...")
                skipped += 1
        
        # Step 3: Combine results - full articles first, then fill with headline-only if needed
        results = full_article_results.copy()
        remaining_slots = max_articles - len(results)
        
        if remaining_slots > 0 and headline_only_results:
            results.extend(headline_only_results[:remaining_slots])
            print(f"📝 Added {min(remaining_slots, len(headline_only_results))} headline-only analyses to fill remaining slots")
        
        print(f"✅ Successfully processed {len(results)} articles ({len(full_article_results)} full, {len(results) - len(full_article_results)} headline-only, {skipped} skipped)")
        
        # Step 4: Generate summary
        summary = self.analyzer.summarize_results(results)
        
        return {
            'summary': summary,
            'articles': results
        }
    
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
    json_path: str = 'sentiment_report.json'
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
    
    Returns:
        dict with 'summary' and 'articles' keys
        
    Example:
        from main_scraper import analyze_news_sync
        
        results = analyze_news_sync(
            keywords=["US economy", "Federal Reserve"],
            max_articles=10
        )
        print(results['summary']['overall_assessment'])
    """
    return asyncio.run(_analyze_news_async(
        keywords, max_articles, print_report, save_json, json_path
    ))


async def analyze_news_async(
    keywords: List[str] = None, 
    max_articles: int = 10,
    print_report: bool = False,
    save_json: bool = False,
    json_path: str = 'sentiment_report.json'
) -> dict:
    """
    Async version for news sentiment analysis.
    Use this if your main app is already async.
    
    Example:
        from main_scraper import analyze_news_async
        
        async def my_main():
            results = await analyze_news_async(keywords=["US economy"])
            print(results['summary']['overall_assessment'])
    """
    return await _analyze_news_async(
        keywords, max_articles, print_report, save_json, json_path
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
    json_path: str = 'sentiment_report.json'
) -> dict:
    """Internal async implementation."""
    if keywords is None:
        keywords = ["US economy"]
    
    analyzer = EconomicNewsSentimentAnalyzer()
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
