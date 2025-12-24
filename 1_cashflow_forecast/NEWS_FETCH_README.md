# News Fetch Server for Cash Flow Dashboard

This directory contains the news fetching functionality for the Cash Flow Forecasting Dashboard.

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install flask flask-cors
   ```

2. **Start the news server**:
   ```bash
   cd 1_cashflow_forecast
   python news_server.py
   ```

3. **Open the dashboard**:
   - Navigate to `http://localhost:5001` in your browser
   - Or open `outputs/dashboards/interactive_dashboard.html` directly

4. **Fetch fresh news**:
   - Go to the "Entity Analysis" tab
   - Enter the max number of articles per country (1-20)
   - Click "Fetch Current News"

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (HTML)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  News Fetch Control Panel                            │   │
│  │  ┌──────────────┐  ┌────────────────────────┐       │   │
│  │  │ Max Articles │  │ Fetch Current News     │       │   │
│  │  │     [5]      │  │        Button          │       │   │
│  │  └──────────────┘  └────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼ HTTP Request                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  News Server (Flask)                        │
│                  http://localhost:5001                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ /api/fetch-news?max_articles=N                      │   │
│  │   - Fetches news for all 8 countries                │   │
│  │   - Updates JSON files in country_sentiments/       │   │
│  │   - Returns news data as JSON                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    News Scraper                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Scrapes Google News RSS                           │   │
│  │ - Extracts article content                          │   │
│  │ - Performs sentiment analysis                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the dashboard HTML |
| `/api/fetch-news` | GET/POST | Fetch fresh news for all countries |
| `/api/news/<country>` | GET | Get cached news for a specific country |
| `/api/news-all` | GET | Get all cached news data |
| `/api/status` | GET | Server status and endpoint list |

### Fetch News Parameters

- `max_articles` (int, 1-20): Maximum articles to fetch per country

**GET Example:**
```
GET http://localhost:5001/api/fetch-news?max_articles=5
```

**POST Example:**
```json
POST http://localhost:5001/api/fetch-news
Content-Type: application/json

{
    "max_articles": 10
}
```

## Countries Covered

| Entity Code | Country | Keywords |
|-------------|---------|----------|
| TH10 | Thailand | Thailand economy, Thai baht |
| TW10 | Taiwan | Taiwan economy, Taiwan dollar |
| SS10 | Singapore | Singapore economy, Singapore dollar |
| MY10 | Malaysia | Malaysia economy, Malaysian ringgit |
| VN20 | Vietnam | Vietnam economy, Vietnamese dong |
| KR10 | South Korea | South Korea economy, Korean won |
| ID10 | Indonesia | Indonesia economy, Indonesian rupiah |
| PH10 | Philippines | Philippines economy, Philippine peso |

## Files

- `news_server.py` - Flask server for handling news fetch requests
- `outputs/dashboards/interactive_dashboard.html` - Generated dashboard with news fetch control
- `../News_scraper/country_sentiments/` - Cached sentiment JSON files

## Troubleshooting

**"Cannot connect to news server"**
- Make sure the server is running: `python news_server.py`
- Check that port 5001 is not in use

**"Import could not be resolved"**
- Install Flask: `pip install flask flask-cors`

**Slow news fetching**
- Fetching news involves web scraping and sentiment analysis
- Each country may take 10-30 seconds depending on network speed
- Consider reducing `max_articles` for faster results
