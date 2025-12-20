from typing import Dict, List

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT (pre-trained for financial text)."""
    
    def __init__(self, use_gpu: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Install transformers: pip install transformers torch")
        
        # Detect device: GPU (cuda) if available, else CPU
        if use_gpu and torch.cuda.is_available():
            self.device = 0  # Use first GPU
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
            print("🚀 Using Apple Silicon GPU (MPS)")
        else:
            self.device = -1  # CPU
            if use_gpu:
                print("⚠️ GPU requested but not available. Using CPU.")
                print("💡 To fix CUDA, try: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("💻 Using CPU")
        
        # FinBERT - specifically trained on financial news
        self.model_name = "ProsusAI/finbert"
        self.classifier = pipeline(
            "sentiment-analysis", 
            model=self.model_name, 
            tokenizer=self.model_name,
            device=self.device
        )
        self.max_length = 512
        self.min_article_length = 200  # Minimum chars to consider "full article"
    
    def analyze(self, article: Dict) -> Dict:
        """Analyze sentiment using FinBERT."""
        title = article.get('title', '')
        text = article.get('text', '')
        
        # Determine if we have full article content or just headline
        has_full_article = len(text) > self.min_article_length and text != title
        
        # Combine title and text, giving priority to actual content
        if has_full_article:
            # Use title + beginning of article (most important info usually at start)
            combined_text = f"{title}. {text}"
            analysis_type = "full_article"
        else:
            # Fallback to just title/headline
            combined_text = title if title else text
            analysis_type = "headline_only"
        
        # Truncate to reasonable length for model (FinBERT handles 512 tokens)
        # ~4 chars per token on average, so 512*4 = 2048 chars
        combined_text = combined_text[:2048]
        
        try:
            result = self.classifier(combined_text, truncation=True, max_length=self.max_length)[0]
            label = result['label'].lower()  # positive, negative, neutral
            score = result['score']
            
            # Convert to -1 to 1 scale
            if label == 'positive':
                final_score = score
                category = "POSITIVE"
                impact = "GOOD for US Economy"
            elif label == 'negative':
                final_score = -score
                category = "NEGATIVE"
                impact = "BAD for US Economy"
            else:
                final_score = 0
                category = "NEUTRAL"
                impact = "MIXED/UNCERTAIN impact"
                
        except Exception as e:
            print(f"FinBERT error: {e}")
            final_score = 0
            score = 0
            category = "NEUTRAL"
            impact = "ANALYSIS ERROR"
            analysis_type = "error"
        
        return {
            'title': article.get('title', 'Unknown'),
            'url': article.get('url', ''),
            'publish_date': article.get('publish_date', None),
            'model': 'FinBERT',
            'final_score': round(final_score, 3),
            'confidence': round(score, 3),
            'category': category,
            'impact': impact,
            'analysis_type': analysis_type,  # 'full_article' or 'headline_only'
            'text_length': len(text)  # For debugging
        }
    
    def summarize_results(self, results: List[Dict]) -> Dict:
        """Summarize overall sentiment from multiple articles."""
        if not results:
            return {'overall': 'NO DATA', 'score': 0}
        
        avg_score = sum(r['final_score'] for r in results) / len(results)
        positive = sum(1 for r in results if r['category'] == 'POSITIVE')
        negative = sum(1 for r in results if r['category'] == 'NEGATIVE')
        neutral = sum(1 for r in results if r['category'] == 'NEUTRAL')
        
        if avg_score > 0.15:
            overall = "BULLISH - Positive economic outlook"
        elif avg_score < -0.15:
            overall = "BEARISH - Negative economic outlook"
        else:
            overall = "NEUTRAL - Mixed signals"
        
        return {
            'model': 'FinBERT',
            'overall_assessment': overall,
            'average_score': round(avg_score, 3),
            'total_articles': len(results),
            'positive_articles': positive,
            'negative_articles': negative,
            'neutral_articles': neutral,
            'sentiment_ratio': f"{positive}:{negative}:{neutral} (pos:neg:neutral)"
        }
