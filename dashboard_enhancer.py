"""
================================================================================
DASHBOARD ENHANCER: NEWS SENTIMENT + CASHFLOW FORECAST INTEGRATION
================================================================================

This module integrates the news sentiment analysis from the News_scraper with
the cashflow forecast data to generate enhanced stakeholder recommendations.

The integration provides:
1. Real-time economic sentiment analysis from news sources
2. Combined forecast + sentiment recommendations
3. Risk-adjusted insights for treasury management
4. Enhanced dashboard with Market Insights section

================================================================================
"""

import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re

# Add News_scraper to path
sys.path.insert(0, str(Path(__file__).parent / "News_scraper"))

try:
    from main_scraper import analyze_news_async, get_full_summary
    NEWS_SCRAPER_AVAILABLE = True
except Exception as e:
    NEWS_SCRAPER_AVAILABLE = False
    print(f"[WARNING] News scraper not available: {e}")


# ==============================================================================
# REGION MAPPING FOR APAC ENTITIES
# ==============================================================================

ENTITY_REGION_MAP = {
    "TW10": {"country": "Taiwan", "region": "East Asia", "currency": "TWD", "keywords": ["Taiwan economy", "Asia Pacific trade"]},
    "PH10": {"country": "Philippines", "region": "Southeast Asia", "currency": "PHP", "keywords": ["Philippines economy", "ASEAN trade"]},
    "TH10": {"country": "Thailand", "region": "Southeast Asia", "currency": "THB", "keywords": ["Thailand economy", "ASEAN trade"]},
    "ID10": {"country": "Indonesia", "region": "Southeast Asia", "currency": "IDR", "keywords": ["Indonesia economy", "ASEAN trade"]},
    "SS10": {"country": "Singapore", "region": "Southeast Asia", "currency": "SGD", "keywords": ["Singapore economy", "Asia Pacific trade"]},
    "MY10": {"country": "Malaysia", "region": "Southeast Asia", "currency": "MYR", "keywords": ["Malaysia economy", "ASEAN trade"]},
    "VN20": {"country": "Vietnam", "region": "Southeast Asia", "currency": "VND", "keywords": ["Vietnam economy", "ASEAN trade"]},
    "KR10": {"country": "South Korea", "region": "East Asia", "currency": "KRW", "keywords": ["South Korea economy", "Asia Pacific trade"]},
}

# Global economic keywords
GLOBAL_KEYWORDS = [
    "global economy outlook",
    "Asia Pacific economy",
    "US Federal Reserve",
    "USD exchange rate",
    "pharmaceutical industry"
]


class SentimentForecastIntegrator:
    """
    Integrates news sentiment analysis with cashflow forecast data
    to generate enhanced recommendations for stakeholders.
    """
    
    def __init__(self, forecast_dir: str = "1_cashflow_forecast"):
        self.forecast_dir = Path(__file__).parent / forecast_dir
        self.processed_dir = self.forecast_dir / "processed_data"
        self.dashboard_dir = self.forecast_dir / "outputs" / "dashboards"
        self.sentiment_cache = {}
        
    async def fetch_sentiment_data(
        self, 
        keywords: List[str] = None,
        max_articles: int = 5,
        use_cache: bool = True
    ) -> Dict:
        """
        Fetch sentiment analysis from news scraper.
        
        Args:
            keywords: Search keywords for news (default: global economic keywords)
            max_articles: Maximum articles to analyze
            use_cache: Whether to use cached sentiment data
            
        Returns:
            Dictionary with sentiment summary and articles
        """
        if not NEWS_SCRAPER_AVAILABLE:
            return self._get_fallback_sentiment()
        
        if keywords is None:
            keywords = GLOBAL_KEYWORDS[:2]  # Use first 2 global keywords
        
        cache_key = "_".join(sorted(keywords))
        if use_cache and cache_key in self.sentiment_cache:
            print(f"[CACHE] Using cached sentiment for: {keywords}")
            return self.sentiment_cache[cache_key]
        
        try:
            print(f"[FETCH] Fetching news sentiment for: {keywords}")
            results = await analyze_news_async(
                keywords=keywords,
                max_articles=max_articles,
                print_report=False,
                save_json=False
            )
            
            summary = get_full_summary(results)
            self.sentiment_cache[cache_key] = summary
            return summary
            
        except Exception as e:
            print(f"[WARNING] Error fetching sentiment: {e}")
            return self._get_fallback_sentiment()
    
    def _get_fallback_sentiment(self) -> Dict:
        """Return fallback sentiment when scraper is unavailable."""
        # Try to load from saved report
        report_path = Path(__file__).parent / "News_scraper" / "sentiment_report.json"
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    data = json.load(f)
                return {
                    'verdict': self._score_to_verdict(data['summary'].get('average_score', 0)),
                    'score': data['summary'].get('average_score', 0),
                    'total_articles': data['summary'].get('total_articles', 0),
                    'articles': data.get('articles', [])
                }
            except:
                pass
        
        return {
            'verdict': 'UNCERTAIN',
            'score': 0.0,
            'total_articles': 0,
            'articles': []
        }
    
    def _score_to_verdict(self, score: float) -> str:
        """Convert numeric score to verdict string."""
        if score > 0.2:
            return "POSITIVE"
        elif score < -0.2:
            return "NEGATIVE"
        return "UNCERTAIN"
    
    def load_forecast_data(self) -> Dict[str, pd.DataFrame]:
        """Load weekly forecast data for all entities."""
        entity_data = {}
        
        for entity in ENTITY_REGION_MAP.keys():
            file_path = self.processed_dir / f"weekly_{entity}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, parse_dates=['Week_Start'])
                    entity_data[entity] = df
                    print(f"  [OK] Loaded {entity}: {len(df)} weeks of data")
                except Exception as e:
                    print(f"  [FAIL] Error loading {entity}: {e}")
        
        return entity_data
    
    def analyze_forecast_trends(self, entity_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Analyze forecast trends for each entity.
        
        Returns dictionary with trend analysis per entity including:
        - Recent trend direction
        - Volatility assessment
        - Growth/decline rate
        """
        trends = {}
        
        for entity, df in entity_data.items():
            # Support both 'Net' and 'Total_Net' column names
            net_col = 'Net' if 'Net' in df.columns else 'Total_Net' if 'Total_Net' in df.columns else None
            if net_col is None or len(df) < 4:
                continue
            
            # Get recent data (last 8 weeks if available)
            recent = df.tail(8)[net_col].values
            
            # Calculate trend
            if len(recent) >= 4:
                first_half = recent[:len(recent)//2].mean()
                second_half = recent[len(recent)//2:].mean()
                
                if second_half > first_half * 1.1:
                    trend_direction = "UPWARD"
                    trend_strength = min((second_half / first_half - 1) * 100, 100)
                elif second_half < first_half * 0.9:
                    trend_direction = "DOWNWARD"
                    trend_strength = min((1 - second_half / first_half) * 100, 100)
                else:
                    trend_direction = "STABLE"
                    trend_strength = 0
            else:
                trend_direction = "INSUFFICIENT_DATA"
                trend_strength = 0
            
            # Calculate volatility (coefficient of variation)
            if recent.std() > 0 and recent.mean() != 0:
                volatility = abs(recent.std() / recent.mean()) * 100
                if volatility > 50:
                    volatility_level = "HIGH"
                elif volatility > 25:
                    volatility_level = "MODERATE"
                else:
                    volatility_level = "LOW"
            else:
                volatility = 0
                volatility_level = "STABLE"
            
            # Net position assessment
            avg_net = recent.mean()
            if avg_net > 0:
                position = "POSITIVE"
            elif avg_net < 0:
                position = "NEGATIVE"
            else:
                position = "NEUTRAL"
            
            trends[entity] = {
                "trend_direction": trend_direction,
                "trend_strength": round(trend_strength, 1),
                "volatility_level": volatility_level,
                "volatility_pct": round(volatility, 1),
                "net_position": position,
                "avg_net_flow": round(avg_net, 2),
                "country": ENTITY_REGION_MAP[entity]["country"],
                "region": ENTITY_REGION_MAP[entity]["region"]
            }
        
        return trends
    
    def generate_recommendations(
        self,
        sentiment_data: Dict,
        forecast_trends: Dict[str, Dict]
    ) -> Dict:
        """
        Generate comprehensive recommendations combining sentiment and forecast data.
        
        The recommendation engine uses a risk matrix approach:
        - Positive sentiment + Positive forecast = Opportunity
        - Negative sentiment + Negative forecast = High Risk / Conservative action
        - Mixed signals = Caution / Monitor closely
        """
        
        recommendations = {
            "overall_outlook": "",
            "sentiment_summary": {
                "verdict": sentiment_data.get("verdict", "UNCERTAIN"),
                "score": sentiment_data.get("score", 0),
                "article_count": sentiment_data.get("total_articles", 0)
            },
            "strategic_recommendations": [],
            "entity_actions": {},
            "risk_assessment": "",
            "key_articles": []
        }
        
        # Overall Market Outlook
        sentiment_score = sentiment_data.get("score", 0)
        sentiment_verdict = sentiment_data.get("verdict", "UNCERTAIN")
        
        # Aggregate forecast trends
        positive_trends = sum(1 for t in forecast_trends.values() if t["trend_direction"] == "UPWARD")
        negative_trends = sum(1 for t in forecast_trends.values() if t["trend_direction"] == "DOWNWARD")
        total_entities = len(forecast_trends)
        
        # Combined assessment matrix
        if sentiment_score > 0.2 and positive_trends > negative_trends:
            recommendations["overall_outlook"] = "FAVORABLE"
            recommendations["risk_assessment"] = "LOW to MODERATE"
            recommendations["strategic_recommendations"] = [
                "Market conditions support growth initiatives in APAC region",
                "Consider accelerating planned investments and expansions",
                "Maintain current liquidity buffers while exploring yield optimization",
                "Review FX hedging strategies to capture favorable conditions"
            ]
        elif sentiment_score < -0.2 and negative_trends > positive_trends:
            recommendations["overall_outlook"] = "CHALLENGING"
            recommendations["risk_assessment"] = "ELEVATED"
            recommendations["strategic_recommendations"] = [
                "Adopt conservative cash management stance",
                "Increase liquidity buffers by 10-15% as precautionary measure",
                "Defer non-essential capital expenditures",
                "Strengthen collection efforts and accelerate receivables",
                "Review and potentially increase FX hedging ratios"
            ]
        elif sentiment_score < -0.2 or negative_trends > positive_trends:
            recommendations["overall_outlook"] = "CAUTIOUS"
            recommendations["risk_assessment"] = "MODERATE to ELEVATED"
            recommendations["strategic_recommendations"] = [
                "Monitor market conditions closely with weekly reviews",
                "Maintain current operational tempo but avoid new commitments",
                "Ensure adequate working capital reserves",
                "Prepare contingency plans for potential market deterioration"
            ]
        else:
            recommendations["overall_outlook"] = "NEUTRAL"
            recommendations["risk_assessment"] = "MODERATE"
            recommendations["strategic_recommendations"] = [
                "Maintain balanced approach to cash management",
                "Continue normal business operations with standard monitoring",
                "Keep strategic reserves at target levels",
                "Stay vigilant for emerging trends in either direction"
            ]
        
        # Entity-specific actions
        for entity, trend in forecast_trends.items():
            entity_recommendation = self._generate_entity_action(
                entity, trend, sentiment_verdict
            )
            recommendations["entity_actions"][entity] = entity_recommendation
        
        # Key articles for reference
        articles = sentiment_data.get("articles", [])
        recommendations["key_articles"] = [
            {
                "title": art.get("title", "Unknown")[:100],
                "sentiment": art.get("category", "NEUTRAL"),
                "score": art.get("score", art.get("final_score", 0)),
                "url": art.get("url", ""),
                "source": art.get("source", "unknown")
            }
            for art in articles[:5]
        ]
        
        return recommendations
    
    def _generate_entity_action(
        self, 
        entity: str, 
        trend: Dict, 
        global_sentiment: str
    ) -> Dict:
        """Generate specific action for an entity based on its trends and global sentiment."""
        
        action = {
            "status": "",
            "priority": "",
            "actions": [],
            "country": trend["country"]
        }
        
        # Risk scoring
        risk_score = 0
        
        if trend["trend_direction"] == "DOWNWARD":
            risk_score += 2
        elif trend["trend_direction"] == "UPWARD":
            risk_score -= 1
        
        if trend["volatility_level"] == "HIGH":
            risk_score += 2
        elif trend["volatility_level"] == "MODERATE":
            risk_score += 1
        
        if trend["net_position"] == "NEGATIVE":
            risk_score += 1
        
        if global_sentiment == "NEGATIVE":
            risk_score += 1
        elif global_sentiment == "POSITIVE":
            risk_score -= 1
        
        # Determine status and priority
        if risk_score >= 4:
            action["status"] = "⚠️ ATTENTION REQUIRED"
            action["priority"] = "HIGH"
            action["actions"] = [
                f"Review {trend['country']} cash position immediately",
                "Consider reducing exposure or increasing reserves",
                "Implement daily monitoring of cash flows"
            ]
        elif risk_score >= 2:
            action["status"] = "👁️ MONITOR CLOSELY"
            action["priority"] = "MEDIUM"
            action["actions"] = [
                f"Increase oversight of {trend['country']} operations",
                "Weekly cash flow reviews recommended",
                "Prepare contingency funding options"
            ]
        elif risk_score <= -1:
            action["status"] = "✅ PERFORMING WELL"
            action["priority"] = "LOW"
            action["actions"] = [
                f"{trend['country']} showing positive momentum",
                "Consider this entity for excess cash deployment",
                "Standard monitoring sufficient"
            ]
        else:
            action["status"] = "📊 STABLE"
            action["priority"] = "STANDARD"
            action["actions"] = [
                f"{trend['country']} operating within normal parameters",
                "Continue standard oversight procedures"
            ]
        
        action["trend"] = trend["trend_direction"]
        action["volatility"] = trend["volatility_level"]
        
        return action
    
    def generate_insights_html(self, recommendations: Dict) -> str:
        """
        Generate HTML section for Market Insights & Recommendations
        to be inserted into the dashboard.
        """
        
        # Determine styling based on outlook
        outlook = recommendations["overall_outlook"]
        if outlook == "FAVORABLE":
            outlook_color = "#27ae60"
            outlook_bg = "linear-gradient(135deg, #d4efdf 0%, #a9dfbf 100%)"
            outlook_icon = "📈"
        elif outlook == "CHALLENGING":
            outlook_color = "#e74c3c"
            outlook_bg = "linear-gradient(135deg, #fadbd8 0%, #f5b7b1 100%)"
            outlook_icon = "📉"
        elif outlook == "CAUTIOUS":
            outlook_color = "#f39c12"
            outlook_bg = "linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%)"
            outlook_icon = "⚠️"
        else:
            outlook_color = "#3498db"
            outlook_bg = "linear-gradient(135deg, #ebf5fb 0%, #d4e6f1 100%)"
            outlook_icon = "📊"
        
        # Build entity status cards
        entity_cards_html = ""
        priority_order = {"HIGH": 1, "MEDIUM": 2, "STANDARD": 3, "LOW": 4}
        sorted_entities = sorted(
            recommendations["entity_actions"].items(),
            key=lambda x: priority_order.get(x[1]["priority"], 5)
        )
        
        for entity, action in sorted_entities:
            if action["priority"] == "HIGH":
                card_border = "#e74c3c"
                card_bg = "#fef9f9"
            elif action["priority"] == "MEDIUM":
                card_border = "#f39c12"
                card_bg = "#fffbf5"
            elif action["priority"] == "LOW":
                card_border = "#27ae60"
                card_bg = "#f5fdf8"
            else:
                card_border = "#3498db"
                card_bg = "#f8fbfe"
            
            entity_cards_html += f'''
            <div style="background: {card_bg}; border-left: 4px solid {card_border}; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="font-size: 1.1rem;">{entity}</strong>
                    <span style="font-size: 0.85rem; padding: 0.2rem 0.6rem; background: {card_border}20; color: {card_border}; border-radius: 12px; font-weight: 500;">{action["priority"]} Priority</span>
                </div>
                <div style="color: #555; font-size: 0.95rem; margin-bottom: 0.5rem;">{action["status"]}</div>
                <div style="font-size: 0.85rem; color: #666;">
                    <span style="margin-right: 1rem;">📍 {action["country"]}</span>
                    <span style="margin-right: 1rem;">📈 {action["trend"]}</span>
                    <span>🔄 {action["volatility"]} volatility</span>
                </div>
            </div>
            '''
        
        # Build recommendations list
        rec_list_html = ""
        for rec in recommendations["strategic_recommendations"]:
            rec_list_html += f'<li style="margin-bottom: 0.6rem; padding-left: 0.5rem;">{rec}</li>'
        
        # Build articles section
        articles_html = ""
        for article in recommendations["key_articles"]:
            if article["sentiment"] == "POSITIVE":
                sent_color = "#27ae60"
                sent_emoji = "🟢"
            elif article["sentiment"] == "NEGATIVE":
                sent_color = "#e74c3c"
                sent_emoji = "🔴"
            else:
                sent_color = "#f39c12"
                sent_emoji = "🟡"
            
            articles_html += f'''
            <div style="padding: 0.8rem 0; border-bottom: 1px solid #eee;">
                <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
                    <span>{sent_emoji}</span>
                    <div style="flex: 1;">
                        <a href="{article["url"]}" target="_blank" style="color: #2c3e50; text-decoration: none; font-weight: 500; font-size: 0.9rem;">{article["title"]}</a>
                        <div style="font-size: 0.8rem; color: #888; margin-top: 0.3rem;">
                            Score: {article["score"]:.2f} | Source: {article["source"]}
                        </div>
                    </div>
                </div>
            </div>
            '''
        
        # Sentiment gauge value (convert -1 to 1 scale to 0-100)
        sentiment_score = recommendations["sentiment_summary"]["score"]
        gauge_value = (sentiment_score + 1) * 50  # Maps -1..1 to 0..100
        
        html = f'''
        <!-- Market Insights & Recommendations Section -->
        <section class="executive-summary" id="market-insights" style="margin-bottom: 2rem;">
            <h2 class="summary-title" style="display: flex; align-items: center; gap: 0.5rem;">
                🌐 Market Insights & Strategic Recommendations
                <span style="font-size: 0.8rem; font-weight: normal; color: #888; margin-left: auto;">
                    Powered by News Sentiment Analysis | Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                </span>
            </h2>
            
            <!-- Overall Outlook Banner -->
            <div style="background: {outlook_bg}; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; border: 1px solid {outlook_color}30;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 3rem;">{outlook_icon}</span>
                    <div>
                        <h3 style="color: {outlook_color}; font-size: 1.5rem; margin: 0;">
                            Overall Outlook: {outlook}
                        </h3>
                        <p style="color: #555; margin: 0.5rem 0 0 0; font-size: 1rem;">
                            Risk Assessment: <strong>{recommendations["risk_assessment"]}</strong> | 
                            Based on {recommendations["sentiment_summary"]["article_count"]} news articles analyzed
                        </p>
                    </div>
                    <div style="margin-left: auto; text-align: center;">
                        <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;">Sentiment Score</div>
                        <div style="font-size: 2rem; font-weight: 700; color: {outlook_color};">
                            {sentiment_score:+.2f}
                        </div>
                        <div style="font-size: 0.75rem; color: #888;">Scale: -1 to +1</div>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem;">
                
                <!-- Strategic Recommendations -->
                <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
                    <h4 style="color: var(--az-mulberry); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        💡 Strategic Recommendations
                    </h4>
                    <ul style="list-style: none; padding: 0; margin: 0; color: #444; line-height: 1.8;">
                        {rec_list_html}
                    </ul>
                </div>
                
                <!-- Entity Priority Matrix -->
                <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
                    <h4 style="color: var(--az-mulberry); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        🎯 Entity Priority Matrix
                    </h4>
                    <div style="max-height: 350px; overflow-y: auto;">
                        {entity_cards_html}
                    </div>
                </div>
            </div>
            
            <!-- News Articles Section -->
            <div style="background: white; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
                <h4 style="color: var(--az-mulberry); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    📰 Key News Articles Analyzed
                    <span style="font-size: 0.8rem; font-weight: normal; color: #888; margin-left: auto;">
                        Click headlines to read full articles
                    </span>
                </h4>
                <div style="columns: 2; column-gap: 2rem;">
                    {articles_html if articles_html else '<p style="color: #888;">No articles available. Run news scraper to fetch latest news.</p>'}
                </div>
            </div>
            
            <!-- Disclaimer -->
            <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; font-size: 0.8rem; color: #666;">
                <strong>⚠️ Disclaimer:</strong> These recommendations are generated using AI-based sentiment analysis of news articles 
                combined with historical cash flow patterns. They should be used as supplementary insights alongside 
                professional judgment and not as the sole basis for financial decisions. Market conditions can change rapidly.
            </div>
        </section>
        '''
        
        return html
    
    def enhance_dashboard(self, recommendations: Dict) -> str:
        """
        Enhance the existing dashboard with Market Insights section.
        
        Returns the path to the enhanced dashboard.
        """
        dashboard_path = self.dashboard_dir / "interactive_dashboard.html"
        
        if not dashboard_path.exists():
            print(f"[ERROR] Dashboard not found at: {dashboard_path}")
            return None
        
        # Read existing dashboard
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # First, remove any existing market insights section to avoid duplicates
        if 'id="market-insights"' in html_content:
            # Find and remove the old section
            old_start = html_content.find('<!-- Market Insights & Recommendations Section -->')
            if old_start == -1:
                old_start = html_content.find('<section class="executive-summary" id="market-insights"')
            if old_start != -1:
                # Find the closing </section> tag for this section
                # Count nested sections to find the right closing tag
                search_pos = old_start
                section_depth = 0
                old_end = -1
                while search_pos < len(html_content):
                    next_open = html_content.find('<section', search_pos)
                    next_close = html_content.find('</section>', search_pos)
                    
                    if next_close == -1:
                        break
                    
                    if next_open != -1 and next_open < next_close:
                        section_depth += 1
                        search_pos = next_open + 8
                    else:
                        if section_depth == 0:
                            old_end = next_close + len('</section>')
                            break
                        section_depth -= 1
                        search_pos = next_close + 10
                
                if old_end != -1:
                    html_content = html_content[:old_start].rstrip() + '\n\n        ' + html_content[old_end:].lstrip()
        
        # Generate insights section
        insights_html = self.generate_insights_html(recommendations)
        
        # Find the correct insertion point: after <!-- Entity Sections --> comment
        # and before the first <div class="entity-section">
        entity_comment = '<!-- Entity Sections -->'
        entity_comment_pos = html_content.find(entity_comment)
        
        if entity_comment_pos != -1:
            # Insert after the comment
            insert_pos = entity_comment_pos + len(entity_comment)
            html_content = (
                html_content[:insert_pos] + 
                '\n\n        ' + insights_html + '\n' +
                html_content[insert_pos:]
            )
        else:
            # Fallback: find first entity-section div in main content
            main_content_pos = html_content.find('<main class="main-content">')
            if main_content_pos != -1:
                first_entity = html_content.find('<div class="entity-section"', main_content_pos)
                if first_entity != -1:
                    html_content = (
                        html_content[:first_entity] + 
                        insights_html + '\n\n        ' +
                        html_content[first_entity:]
                    )
        
        # Save enhanced dashboard
        enhanced_path = self.dashboard_dir / "interactive_dashboard.html"
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[OK] Dashboard enhanced with Market Insights section")
        return str(enhanced_path)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

async def main():
    """
    Main execution function for the dashboard enhancer.
    
    This function:
    1. Loads cashflow forecast data
    2. Fetches news sentiment analysis
    3. Generates combined recommendations
    4. Enhances the dashboard with insights section
    """
    print("=" * 70)
    print("  DASHBOARD ENHANCER")
    print("  Integrating News Sentiment with Cashflow Forecasts")
    print("=" * 70)
    
    integrator = SentimentForecastIntegrator()
    
    # Step 1: Load forecast data
    print("\n[Step 1] Loading forecast data...")
    entity_data = integrator.load_forecast_data()
    
    if not entity_data:
        print("[ERROR] No forecast data found. Run the cashflow pipeline first.")
        return
    
    # Step 2: Analyze forecast trends
    print("\n[Step 2] Analyzing forecast trends...")
    forecast_trends = integrator.analyze_forecast_trends(entity_data)
    for entity, trend in forecast_trends.items():
        print(f"  {entity} ({trend['country']}): {trend['trend_direction']} | {trend['volatility_level']} volatility")
    
    # Step 3: Fetch sentiment data
    print("\n[Step 3] Fetching news sentiment analysis...")
    sentiment_data = await integrator.fetch_sentiment_data(
        keywords=["US economy", "Asia Pacific economy"],
        max_articles=5
    )
    print(f"  Verdict: {sentiment_data.get('verdict', 'N/A')}")
    print(f"  Score: {sentiment_data.get('score', 0):.3f}")
    print(f"  Articles: {sentiment_data.get('total_articles', 0)}")
    
    # Step 4: Generate recommendations
    print("\n[Step 4] Generating strategic recommendations...")
    recommendations = integrator.generate_recommendations(
        sentiment_data,
        forecast_trends
    )
    print(f"  Overall Outlook: {recommendations['overall_outlook']}")
    print(f"  Risk Assessment: {recommendations['risk_assessment']}")
    
    # Step 5: Enhance dashboard
    print("\n[Step 5] Enhancing dashboard with Market Insights...")
    enhanced_path = integrator.enhance_dashboard(recommendations)
    
    if enhanced_path:
        print(f"\n[SUCCESS] Dashboard enhanced successfully!")
        print(f"   Open: {enhanced_path}")
    
    print("\n" + "=" * 70)
    print("  ENHANCEMENT COMPLETE!")
    print("=" * 70)
    
    return recommendations


def run():
    """Synchronous entry point."""
    return asyncio.run(main())


if __name__ == "__main__":
    run()

