"""
External Context Agent
======================

Integrates external context from the news scraper to influence
detection thresholds and provide additional signals.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

from .base_agent import BaseDetectorAgent
from ..core.models import (
    AnomalyFlag, Feedback, DetectionContext, AnomalyType, Severity
)
from ..core.rule_graph import RuleGraph, RuleNode, NodeType, EdgeType, ThresholdConfig
from ..core.knowledge_base import KnowledgeBase

# Add News_scraper to path
NEWS_SCRAPER_PATH = Path(__file__).parent.parent.parent / 'News_scraper'
if str(NEWS_SCRAPER_PATH) not in sys.path:
    sys.path.insert(0, str(NEWS_SCRAPER_PATH))


class ExternalContextAgent(BaseDetectorAgent):
    """
    Agent for external context integration.
    
    Wraps the news scraper to:
    - Fetch real-time economic news sentiment
    - Map sentiment to detection thresholds
    - Identify major economic events
    - Provide context signals to other agents
    """
    
    def __init__(self, agent_id: str = None, name: str = "External Context Agent",
                 knowledge_base: KnowledgeBase = None,
                 cache_duration_hours: int = 4,
                 default_keywords: List[str] = None):
        """
        Initialize the external context agent.
        
        Args:
            agent_id: Unique identifier
            name: Agent name
            knowledge_base: Shared knowledge base
            cache_duration_hours: How long to cache sentiment results
            default_keywords: Default keywords for news search
        """
        super().__init__(agent_id, name, knowledge_base)
        
        self.config = {
            'cache_duration_hours': cache_duration_hours,
            'max_articles': 5,
            'threshold_modifiers': {
                'POSITIVE': 1.1,   # More lenient thresholds
                'NEUTRAL': 1.0,    # Normal thresholds
                'NEGATIVE': 0.85,  # More sensitive thresholds (lower threshold = more flags)
                'BEARISH': 0.80    # Even more sensitive
            },
            'stress_levels': {
                'HIGH': {'min_score': -0.5, 'max_score': -1.0},
                'ELEVATED': {'min_score': -0.2, 'max_score': -0.5},
                'NORMAL': {'min_score': -0.2, 'max_score': 0.2},
                'LOW': {'min_score': 0.2, 'max_score': 1.0}
            }
        }
        
        self.default_keywords = default_keywords or [
            "US economy", "Federal Reserve", "interest rates",
            "economic outlook", "market volatility"
        ]
        
        # Cached context
        self._cached_context: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        
        # News scraper availability
        self._scraper_available = False
        self._init_scraper()
        
        # Initialize rule graph
        self._init_rule_graphs()
    
    def _init_scraper(self) -> None:
        """Initialize the news scraper if available."""
        try:
            from main_scraper import analyze_news_sync, get_sentiment_verdict, get_sentiment_score
            self._analyze_news_sync = analyze_news_sync
            self._get_sentiment_verdict = get_sentiment_verdict
            self._get_sentiment_score = get_sentiment_score
            self._scraper_available = True
        except ImportError as e:
            print(f"Warning: News scraper not available: {e}")
            self._scraper_available = False
    
    @property
    def agent_type(self) -> str:
        return "external_context"
    
    @property
    def anomaly_types(self) -> List[AnomalyType]:
        return [AnomalyType.EXTERNAL]
    
    def _init_rule_graphs(self) -> None:
        """Initialize context evaluation rule graph."""
        graph = RuleGraph(
            graph_id="ext_context",
            name="External Context Evaluation",
            description="Evaluates external signals for anomaly detection context"
        )
        
        # Check high market stress
        high_stress = RuleNode(
            node_id="high_stress",
            node_type=NodeType.THRESHOLD,
            name="High Market Stress",
            description="Check for high market stress from negative sentiment",
            condition="sentiment_score < -0.5",
            field_name="sentiment_score",
            operator="<",
            threshold=ThresholdConfig(base=-0.5)
        )
        graph.add_node(high_stress, is_root=True)
        
        # Flag high stress context
        flag_stress = RuleNode(
            node_id="flag_high_stress",
            node_type=NodeType.ACTION,
            name="High Stress Alert",
            description="Market stress is high - increase sensitivity",
            condition="High stress",
            action="alert_context",
            severity="high"
        )
        graph.add_node(flag_stress)
        
        # Check elevated stress
        elevated_stress = RuleNode(
            node_id="elevated_stress",
            node_type=NodeType.THRESHOLD,
            name="Elevated Market Stress",
            description="Check for elevated market stress",
            condition="sentiment_score < -0.2",
            field_name="sentiment_score",
            operator="<",
            threshold=ThresholdConfig(base=-0.2)
        )
        graph.add_node(elevated_stress)
        
        # Flag elevated stress
        flag_elevated = RuleNode(
            node_id="flag_elevated",
            node_type=NodeType.ACTION,
            name="Elevated Stress Notice",
            description="Market stress is elevated",
            condition="Elevated stress",
            action="notice_context",
            severity="medium"
        )
        graph.add_node(flag_elevated)
        
        # Normal context
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal Context",
            description="Market conditions appear normal",
            condition="Normal",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("high_stress", "flag_high_stress", EdgeType.ESCALATE)
        graph.add_edge("high_stress", "elevated_stress", EdgeType.ELSE)
        graph.add_edge("elevated_stress", "flag_elevated", EdgeType.ESCALATE)
        graph.add_edge("elevated_stress", "normal", EdgeType.ELSE)
        
        self.rule_graphs['context'] = graph
    
    def fetch_context(self, keywords: List[str] = None, 
                      force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch external context from news scraper.
        
        Args:
            keywords: Search keywords (uses defaults if not provided)
            force_refresh: Force refresh even if cache is valid
        
        Returns:
            Context dictionary with sentiment and signals
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            return self._cached_context
        
        if not self._scraper_available:
            return self._get_default_context()
        
        keywords = keywords or self.default_keywords
        
        try:
            # Fetch news sentiment
            results = self._analyze_news_sync(
                keywords=keywords[:3],  # Limit keywords
                max_articles=self.config['max_articles'],
                print_report=False,
                save_json=False
            )
            
            sentiment_score = self._get_sentiment_score(results)
            verdict = self._get_sentiment_verdict(results)
            
            # Build context
            context = {
                'sentiment_score': sentiment_score,
                'verdict': verdict,
                'market_stress': self._score_to_stress_level(sentiment_score),
                'threshold_modifier': self._get_threshold_modifier(verdict),
                'articles_analyzed': len(results.get('articles', [])),
                'summary': results.get('summary', {}),
                'fetched_at': datetime.now().isoformat(),
                'keywords': keywords[:3]
            }
            
            # Extract key events from articles
            context['recent_events'] = self._extract_events(results.get('articles', []))
            
            # Cache the context
            self._cached_context = context
            self._cache_timestamp = datetime.now()
            
            # Save to knowledge base
            if self.kb:
                self.kb.save_external_context(
                    'news_sentiment',
                    context,
                    valid_hours=self.config['cache_duration_hours']
                )
            
            return context
            
        except Exception as e:
            print(f"Error fetching news context: {e}")
            return self._get_default_context()
    
    def _is_cache_valid(self) -> bool:
        """Check if cached context is still valid."""
        if self._cached_context is None or self._cache_timestamp is None:
            return False
        
        age = datetime.now() - self._cache_timestamp
        max_age = timedelta(hours=self.config['cache_duration_hours'])
        return age < max_age
    
    def _get_default_context(self) -> Dict[str, Any]:
        """Get default context when scraper is unavailable."""
        return {
            'sentiment_score': 0.0,
            'verdict': 'UNCERTAIN',
            'market_stress': 'NORMAL',
            'threshold_modifier': 1.0,
            'articles_analyzed': 0,
            'summary': {},
            'fetched_at': datetime.now().isoformat(),
            'recent_events': [],
            'source': 'default'
        }
    
    def _score_to_stress_level(self, score: float) -> str:
        """Convert sentiment score to market stress level."""
        if score <= -0.5:
            return "HIGH"
        elif score <= -0.2:
            return "ELEVATED"
        elif score >= 0.2:
            return "LOW"
        else:
            return "NORMAL"
    
    def _get_threshold_modifier(self, verdict: str) -> float:
        """Get threshold modifier based on sentiment verdict."""
        modifiers = self.config['threshold_modifiers']
        
        if verdict in ['NEGATIVE', 'BEARISH']:
            return modifiers.get('NEGATIVE', 0.85)
        elif verdict in ['POSITIVE', 'BULLISH']:
            return modifiers.get('POSITIVE', 1.1)
        else:
            return modifiers.get('NEUTRAL', 1.0)
    
    def _extract_events(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key events from articles."""
        events = []
        
        # Keywords that indicate important events
        event_keywords = [
            'federal reserve', 'fed', 'interest rate', 'inflation',
            'recession', 'growth', 'employment', 'jobs', 'gdp',
            'trade', 'tariff', 'regulation', 'banking', 'crisis'
        ]
        
        for article in articles:
            title = article.get('title', '').lower()
            
            # Check for event keywords
            matched_keywords = [kw for kw in event_keywords if kw in title]
            
            if matched_keywords:
                events.append({
                    'title': article.get('title', ''),
                    'sentiment': article.get('category', 'NEUTRAL'),
                    'score': article.get('final_score', 0),
                    'keywords': matched_keywords,
                    'source': article.get('source', 'unknown'),
                    'date': article.get('publish_date')
                })
        
        return events[:5]  # Limit to 5 most relevant events
    
    def get_detection_context(self, entity: str = None, 
                               timestamp: datetime = None) -> DetectionContext:
        """
        Get a DetectionContext object for use by other agents.
        
        Args:
            entity: The entity being analyzed
            timestamp: The timestamp of the data
        
        Returns:
            DetectionContext with external signals
        """
        external = self.fetch_context()
        
        return DetectionContext(
            entity=entity or 'unknown',
            timestamp=timestamp or datetime.now(),
            market_stress=external.get('market_stress', 'NORMAL'),
            sentiment_score=external.get('sentiment_score', 0.0),
            threshold_modifier=external.get('threshold_modifier', 1.0),
            recent_events=external.get('recent_events', [])
        )
    
    def detect(self, data, context: DetectionContext = None) -> List[AnomalyFlag]:
        """
        The external context agent doesn't detect anomalies directly.
        It provides context to other agents. This method returns
        context-based alerts if sentiment is very negative.
        """
        flags = []
        
        # Fetch latest context
        external = self.fetch_context()
        
        # Only create flags for significant market stress
        if external.get('market_stress') in ['HIGH', 'ELEVATED']:
            sentiment = external.get('sentiment_score', 0)
            
            flag = self.create_flag(
                entity=context.entity if context else 'global',
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.EXTERNAL,
                severity=Severity.HIGH if external['market_stress'] == 'HIGH' else Severity.MEDIUM,
                confidence=0.7,
                metric_name="market_stress",
                metric_value=sentiment,
                threshold=-0.2,
                description=f"Market stress: {external['market_stress']} (sentiment: {sentiment:.2f})",
                explanation=f"External economic news indicates {external['market_stress'].lower()} "
                           f"market stress. Sentiment score: {sentiment:.2f}. "
                           f"Detection thresholds have been adjusted by {external['threshold_modifier']:.0%}.",
                rule_id="ext_context",
                contributing_factors={
                    'verdict': external.get('verdict'),
                    'articles_analyzed': external.get('articles_analyzed', 0),
                    'recent_events': [e.get('title') for e in external.get('recent_events', [])]
                }
            )
            flags.append(flag)
        
        return flags
    
    def explain(self, flag: AnomalyFlag) -> str:
        """Generate explanation for external context alert."""
        lines = [
            f"External Context Alert",
            f"======================",
            f"Time: {flag.timestamp}",
            f"Market Stress: {flag.contributing_factors.get('verdict', 'Unknown')}",
            f"Sentiment Score: {flag.metric_value:.2f}",
            f"",
            flag.explanation,
            "",
            "Recent Events:"
        ]
        
        for event in flag.contributing_factors.get('recent_events', []):
            lines.append(f"  • {event}")
        
        return "\n".join(lines)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current external context."""
        context = self.fetch_context()
        
        return {
            'market_stress': context.get('market_stress', 'UNKNOWN'),
            'sentiment': {
                'score': context.get('sentiment_score', 0),
                'verdict': context.get('verdict', 'UNCERTAIN')
            },
            'threshold_modifier': context.get('threshold_modifier', 1.0),
            'data_freshness': self._cache_timestamp.isoformat() if self._cache_timestamp else 'Never fetched',
            'scraper_available': self._scraper_available,
            'recent_events': context.get('recent_events', []),
            'articles_analyzed': context.get('articles_analyzed', 0)
        }
    
    def adjust_thresholds_for_context(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust thresholds based on current external context.
        
        Args:
            base_thresholds: Dictionary of base threshold values
        
        Returns:
            Adjusted thresholds
        """
        context = self.fetch_context()
        modifier = context.get('threshold_modifier', 1.0)
        
        return {
            name: value * modifier 
            for name, value in base_thresholds.items()
        }

