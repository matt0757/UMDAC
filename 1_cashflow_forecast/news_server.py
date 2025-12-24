"""
================================================================================
NEWS FETCH SERVER FOR CASH FLOW DASHBOARD
================================================================================

This lightweight Flask server provides an API endpoint to fetch fresh news
for the Cash Flow Forecasting Dashboard. It allows users to dynamically
update the Market News section without regenerating the entire dashboard.

Usage:
    python news_server.py

The server will run on http://localhost:5001 and provide:
    - GET /api/fetch-news?max_articles=10 - Fetch news for all countries
    - GET /api/news/<country> - Get cached news for a specific country
    - GET / - Redirect to the dashboard HTML

================================================================================
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_file, redirect
from flask_cors import CORS

# Add parent directory to path to import News_scraper modules
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "News_scraper"))

from main_scraper import analyze_news_sync, get_full_summary

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Configuration
DASHBOARD_PATH = SCRIPT_DIR / "outputs" / "dashboards" / "interactive_dashboard.html"
COUNTRY_SENTIMENTS_DIR = PROJECT_ROOT / "News_scraper" / "country_sentiments"

# Country to entity mapping
ENTITY_COUNTRY_MAP = {
    "TH10": {"country": "Thailand", "keywords": ["Thailand economy"]},
    "TW10": {"country": "Taiwan", "keywords": ["Taiwan economy"]},
    "SS10": {"country": "Singapore", "keywords": ["Singapore economy"]},
    "MY10": {"country": "Malaysia", "keywords": ["Malaysia economy"]},
    "VN20": {"country": "Vietnam", "keywords": ["Vietnam economy"]},
    "KR10": {"country": "South Korea", "keywords": ["South Korea economy"]},
    "ID10": {"country": "Indonesia", "keywords": ["Indonesia economy"]},
    "PH10": {"country": "Philippines", "keywords": ["Philippines economy"]},
}

# Reverse mapping: country name to file name
COUNTRY_FILE_MAP = {
    "Thailand": "thailand_sentiment.json",
    "Taiwan": "taiwan_sentiment.json",
    "Singapore": "singapore_sentiment.json",
    "Malaysia": "malaysia_sentiment.json",
    "Vietnam": "vietnam_sentiment.json",
    "South Korea": "south_korea_sentiment.json",
    "Indonesia": "indonesia_sentiment.json",
    "Philippines": "philippines_sentiment.json",
}


def save_country_sentiment(country: str, data: dict) -> bool:
    """Save sentiment data for a country to JSON file."""
    try:
        filename = COUNTRY_FILE_MAP.get(country)
        if not filename:
            print(f"⚠️ No file mapping for country: {country}")
            return False
        
        filepath = COUNTRY_SENTIMENTS_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved sentiment data for {country} to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving sentiment for {country}: {e}")
        return False


def load_country_sentiment(country: str) -> dict:
    """Load cached sentiment data for a country."""
    try:
        filename = COUNTRY_FILE_MAP.get(country)
        if not filename:
            return {"error": f"No file mapping for country: {country}"}
        
        filepath = COUNTRY_SENTIMENTS_DIR / filename
        if not filepath.exists():
            return {"error": f"No cached data for {country}"}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


async def fetch_news_for_country(country: str, keywords: list, max_articles: int) -> dict:
    """Fetch news for a single country asynchronously."""
    print(f"🔍 Fetching news for {country}...")
    try:
        results = analyze_news_sync(
            keywords=keywords,
            max_articles=max_articles,
            print_report=False,
            save_json=False
        )
        
        summary = get_full_summary(results)
        
        # Format the data
        country_data = {
            "country": country,
            "average_score": summary.get('score', 0),
            "total_articles": summary.get('total_articles', 0),
            "articles": summary.get('articles', []),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save to file
        save_country_sentiment(country, country_data)
        
        return country_data
    except Exception as e:
        print(f"❌ Error fetching news for {country}: {e}")
        return {
            "country": country,
            "error": str(e),
            "average_score": 0,
            "total_articles": 0,
            "articles": []
        }


def fetch_all_news(max_articles: int = 5) -> dict:
    """Fetch news for all countries synchronously."""
    results = {}
    
    for entity, config in ENTITY_COUNTRY_MAP.items():
        country = config["country"]
        keywords = config["keywords"]
        
        print(f"\n{'='*50}")
        print(f"Fetching news for {country} ({entity})")
        print(f"Keywords: {keywords}")
        print(f"{'='*50}")
        
        try:
            news_results = analyze_news_sync(
                keywords=keywords,
                max_articles=max_articles,
                print_report=False,
                save_json=False
            )
            
            summary = get_full_summary(news_results)
            
            country_data = {
                "country": country,
                "entity": entity,
                "average_score": summary.get('score', 0),
                "total_articles": summary.get('total_articles', 0),
                "articles": summary.get('articles', []),
                "last_updated": datetime.now().isoformat()
            }
            
            # Save to file
            save_country_sentiment(country, country_data)
            results[entity] = country_data
            
            print(f"✅ {country}: {country_data['total_articles']} articles, score: {country_data['average_score']:.2f}")
            
        except Exception as e:
            print(f"❌ Error fetching news for {country}: {e}")
            results[entity] = {
                "country": country,
                "entity": entity,
                "error": str(e),
                "average_score": 0,
                "total_articles": 0,
                "articles": []
            }
    
    return results


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def index():
    """Serve the dashboard HTML file."""
    if DASHBOARD_PATH.exists():
        return send_file(str(DASHBOARD_PATH))
    return jsonify({"error": "Dashboard not found. Please run the pipeline first."}), 404


@app.route('/api/fetch-news', methods=['GET', 'POST'])
def api_fetch_news():
    """
    Fetch fresh news for all countries.
    
    Query Parameters:
        max_articles (int): Maximum number of articles per country (default: 5)
    
    Returns:
        JSON with news data for all countries
    """
    try:
        # Get max_articles from query params or request body
        if request.method == 'POST':
            data = request.get_json() or {}
            max_articles = data.get('max_articles', 5)
        else:
            max_articles = request.args.get('max_articles', 5, type=int)
        
        # Clamp to reasonable range
        max_articles = max(1, min(20, max_articles))
        
        print(f"\n🚀 Starting news fetch with max_articles={max_articles}")
        
        results = fetch_all_news(max_articles=max_articles)
        
        return jsonify({
            "success": True,
            "message": f"Fetched news for {len(results)} countries",
            "max_articles": max_articles,
            "timestamp": datetime.now().isoformat(),
            "data": results
        })
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/news/<country>', methods=['GET'])
def api_get_country_news(country: str):
    """
    Get cached news for a specific country.
    
    Path Parameters:
        country (str): Country name (e.g., "Thailand", "Taiwan")
    
    Returns:
        JSON with cached news data
    """
    try:
        data = load_country_sentiment(country)
        if "error" in data:
            return jsonify(data), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/news-all', methods=['GET'])
def api_get_all_news():
    """
    Get cached news for all countries.
    
    Returns:
        JSON with all cached news data
    """
    try:
        all_data = {}
        for entity, config in ENTITY_COUNTRY_MAP.items():
            country = config["country"]
            data = load_country_sentiment(country)
            all_data[entity] = data
        
        return jsonify({
            "success": True,
            "data": all_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get server status and available endpoints."""
    return jsonify({
        "status": "running",
        "endpoints": {
            "GET /": "Serve the dashboard HTML",
            "GET /api/fetch-news?max_articles=N": "Fetch fresh news (N articles per country)",
            "POST /api/fetch-news": "Fetch fresh news (send {max_articles: N} in body)",
            "GET /api/news/<country>": "Get cached news for a country",
            "GET /api/news-all": "Get all cached news",
            "GET /api/status": "Get server status"
        },
        "entities": list(ENTITY_COUNTRY_MAP.keys()),
        "countries": list(COUNTRY_FILE_MAP.keys())
    })


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("📰 NEWS FETCH SERVER FOR CASH FLOW DASHBOARD")
    print("=" * 60)
    print(f"Dashboard path: {DASHBOARD_PATH}")
    print(f"Country sentiments dir: {COUNTRY_SENTIMENTS_DIR}")
    print("\nAvailable endpoints:")
    print("  GET  /                    - View dashboard")
    print("  GET  /api/fetch-news      - Fetch fresh news")
    print("  POST /api/fetch-news      - Fetch fresh news (with body)")
    print("  GET  /api/news/<country>  - Get cached country news")
    print("  GET  /api/news-all        - Get all cached news")
    print("  GET  /api/status          - Server status")
    print("\n" + "=" * 60)
    print("Starting server on http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
