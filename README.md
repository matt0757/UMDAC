# 🏦 UMDAC - Unified Multi-agent Detection and Cash Flow Forecasting

<p align="center">
  <strong>AstraZeneca DATATHON 2025 Solution</strong><br>
  An enterprise-grade cash flow forecasting and anomaly detection system for APAC treasury operations
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Core Modules](#-core-modules)
  - [Cash Flow Forecasting Pipeline](#1-cash-flow-forecasting-pipeline)
  - [Multi-Agent Anomaly Detection](#2-multi-agent-anomaly-detection)
  - [News Scraper & Sentiment Analysis](#3-news-scraper--sentiment-analysis)
  - [News Fetch API Server](#4-news-fetch-api-server)
- [Data Sources](#-data-sources)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Performance Metrics](#-performance-metrics)
- [License](#-license)

---

## 🎯 Overview

**UMDAC** is a comprehensive financial analytics platform designed for treasury management across AstraZeneca's Asia-Pacific operations. The system integrates three powerful capabilities:

| Capability | Description |
|------------|-------------|
| **📈 Cash Flow Forecasting** | ML-powered 1-month and 6-month cash flow predictions with multi-model ensemble |
| **🔍 Anomaly Detection** | Multi-agent system for detecting unusual financial transactions and patterns |
| **📰 Market Intelligence** | Real-time news sentiment analysis for enhanced decision-making |

### Key Features

- ✅ **Multi-Entity Support**: Processes 8 APAC entities (TW10, PH10, TH10, ID10, SS10, MY10, VN20, KR10)
- ✅ **Ensemble ML Models**: XGBoost, LightGBM, RandomForest, GradientBoosting, Ridge
- ✅ **Interactive Dashboards**: Plotly.js-powered visualizations with AstraZeneca branding
- ✅ **Real-time News Sentiment**: FinBERT-powered financial news analysis
- ✅ **Self-evolving Rules**: Anomaly detection rules that adapt based on feedback

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              UMDAC SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐     │
│   │   Raw Data      │───▶│  Cash Flow Pipeline │───▶│  Interactive        │     │
│   │   (CSV Files)   │    │  (ML Forecasting)   │    │  Dashboard (HTML)   │     │
│   └─────────────────┘    └─────────────────────┘    └──────────┬──────────┘     │
│                                                                 │                │
│   ┌─────────────────┐    ┌─────────────────────┐                │                │
│   │  Weekly Feature │───▶│  Multi-Agent        │───▶│  Anomaly  │                │
│   │  Data           │    │  Anomaly Detection  │    │  Reports  │                │
│   └─────────────────┘    └─────────────────────┘    └───────────┘                │
│                                                                 │                │
│   ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐      │
│   │  News Sources   │───▶│  News Scraper &     │───▶│  News Fetch API     │      │
│   │  (RSS/Web)      │    │  Sentiment Analysis │    │  (Flask Server)     │      │
│   └─────────────────┘    └─────────────────────┘    └─────────────────────┘      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Core Modules

### 1. Cash Flow Forecasting Pipeline

**Location:** `1_cashflow_forecast/`

A complete end-to-end machine learning pipeline for cash flow forecasting.

#### Components

| File | Purpose |
|------|---------|
| `run_full_pipeline.py` | Main executable - runs entire pipeline from raw data to dashboard |
| `news_server.py` | Flask API server for fetching news and serving dashboard |
| `PIPELINE_DOCUMENTATION.md` | Comprehensive technical documentation |
| `NEWS_FETCH_README.md` | Documentation for the news fetch API server |
| `processed_data/` | Cleaned and feature-engineered weekly data |
| `outputs/dashboards/` | Generated interactive HTML dashboards |

#### Pipeline Flow

```
DataCleaner → WeeklyAggregator → MLForecaster → InteractiveDashboardBuilder
```

#### Key Features

- **Data Cleaning**: Automatic type conversion, missing value handling, category standardization
- **Feature Engineering**: 111+ features including temporal, lag, rolling statistics, and category-based
- **Forward Feature Selection**: Data-driven selection of most predictive features
- **Multi-Model Ensemble**: 5 models with inverse-RMSE weighted averaging
- **Iterative Forecasting**: "Every Year Rhymes" approach for 6-month predictions

#### Model Configuration

| Model | Parameters | Purpose |
|-------|------------|---------|
| XGBoost | n_estimators=200, max_depth=4 | Best-in-class gradient boosting |
| LightGBM | n_estimators=200, max_depth=5 | Fast gradient boosting |
| RandomForest | n_estimators=300, max_depth=8 | Robust bagging ensemble |
| GradientBoosting | n_estimators=250, max_depth=4 | Stable baseline |
| Ridge | alpha=1.0 | Linear baseline with regularization |

#### Forecast Outputs

- **Backtest Validation**: Last 4 weeks actual vs predicted
- **Short-Term Forecast**: 4-week tactical projections
- **Long-Term Forecast**: 24-week (6-month) strategic outlook

---

### 2. Multi-Agent Anomaly Detection

**Location:** `multi_agent_anomaly_detection/`

A sophisticated multi-agent system for detecting financial anomalies with interpretable explanations.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      META-COORDINATOR                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Statistical │ │   Pattern   │ │    Rule     │ │  Temporal   │ │
│  │   Agent     │ │   Agent     │ │   Agent     │ │   Agent     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│                                                                   │
│  ┌─────────────┐                                                 │
│  │  Category   │              ──► Ensemble Verdict               │
│  │   Agent     │                                                 │
│  └─────────────┘                                                 │
└──────────────────────────────────────────────────────────────────┘
```

#### Directory Structure

```
multi_agent_anomaly_detection/
├── agents/                   # Detection agents
│   ├── base_agent.py        # Abstract base class
│   ├── statistical_agent.py # Z-score & deviation detection
│   ├── pattern_agent.py     # Pattern-based detection
│   ├── rule_agent.py        # Business rule validation
│   ├── temporal_agent.py    # Time-based anomalies
│   └── category_agent.py    # Category-specific detection
├── coordination/
│   └── meta_coordinator.py  # Agent orchestration & conflict resolution
├── core/
│   ├── models.py            # Data models (AnomalyFlag, Verdict, etc.)
│   ├── knowledge_base.py    # Persistent storage
│   ├── rule_graph.py        # Rule graph structures
│   └── interpretable_tree.py
├── evolution/
│   ├── rule_evolution.py    # Rule mutation & optimization
│   └── feedback.py          # Feedback collection & performance tracking
├── rules/                    # Rule definitions (JSON)
│   ├── business_rules.json
│   ├── statistical_rules.json
│   └── temporal_rules.json
├── data/
│   └── knowledge_base.db    # SQLite database
├── outputs/
│   ├── dashboards/          # Generated anomaly detection dashboards
│   └── reports/             # JSON reports (detailed_verdicts.json, detection_summary.json)
├── utils/
│   └── helpers.py           # Utility functions
├── run_detection.py         # CLI entry point for detection
└── run_full_pipeline.py     # End-to-end pipeline with dashboard generation
```

#### Agent Types

| Agent | Detection Focus | Anomaly Types |
|-------|-----------------|---------------|
| **Statistical** | Z-scores, distribution outliers | Unusual values beyond expected range |
| **Pattern** | Historical patterns, seasonality | Deviations from established patterns |
| **Rule** | Business rules, thresholds | Violations of predefined limits |
| **Temporal** | Time-based patterns | Unusual timing, sequences |
| **Category** | Category-specific norms | Abnormal transaction categories |

#### Key Features

- **Ensemble Voting**: Multiple agents vote on anomalies with confidence weighting
- **Conflict Resolution**: Intelligent handling when agents disagree
- **Interpretable Explanations**: Clear decision paths for each detection
- **Rule Evolution**: Automatic rule optimization based on feedback
- **Performance Tracking**: Precision, recall, F1 monitoring with degradation alerts

#### Usage

**CLI Interface (run_detection.py):**
```bash
# Run detection on data
python run_detection.py --data weekly_features.csv --entity ID10

# Generate HTML report
python run_detection.py --data data.csv --output report.html --format html

# Generate JSON report
python run_detection.py --data data.csv --output report.json --format json

# Check system status
python run_detection.py --status
```

**Full Pipeline (run_full_pipeline.py):**
```bash
# Run end-to-end pipeline with dashboard generation
python run_full_pipeline.py

# With custom options
python run_full_pipeline.py --data-dir path/to/data --output-dir path/to/output --entities ID10,TW10
```

---

### 3. News Scraper & Sentiment Analysis

**Location:** `News_scraper/`

Automated economic news scraping and sentiment analysis using FinBERT.

#### Components

| File | Purpose |
|------|---------|
| `main_scraper.py` | Main orchestrator with public API |
| `news_scraper.py` | News source scraping (RSS + web) |
| `article_extractor.py` | Full article content extraction |
| `sentiment_analyzer.py` | FinBERT-powered sentiment analysis |
| `sentiment_report.json` | Latest sentiment analysis results |

#### News Sources

- **RSS Feeds** (Primary - no captcha):
  - Google News RSS
  - BBC Business/World
  - CNBC
  - MarketWatch
  - Yahoo Finance

- **Web Scraping** (Fallback with stealth mode):
  - AP News
  - Direct article extraction

#### Sentiment Analysis

Uses **FinBERT** (ProsusAI/finbert) - a BERT model fine-tuned on financial text:

```python
# Sentiment Categories
POSITIVE  → Score > 0.2  → "BULLISH - Positive economic outlook"
NEGATIVE  → Score < -0.2 → "BEARISH - Negative economic outlook"
NEUTRAL   → Otherwise    → "NEUTRAL - Mixed signals"
```

#### API Usage

```python
from main_scraper import analyze_news_sync, get_full_summary

# Synchronous analysis
results = analyze_news_sync(
    keywords=["US economy", "Federal Reserve"],
    max_articles=10
)

# Get summary with verdict
summary = get_full_summary(results)
print(summary['verdict'])  # "POSITIVE", "NEGATIVE", or "UNCERTAIN"
print(summary['score'])    # -1.0 to 1.0
```

---

### 4. News Fetch API Server

**Location:** `1_cashflow_forecast/news_server.py`

A lightweight Flask server that provides REST API endpoints for fetching fresh news and sentiment data for the Cash Flow Forecasting Dashboard. Allows dynamic updates to the Market News section without regenerating the entire dashboard.

#### Features

- **RESTful API**: Simple HTTP endpoints for news fetching
- **Country-Specific News**: Fetches news for all 8 APAC entities
- **Caching**: Stores sentiment data in JSON files for quick access
- **Dashboard Integration**: Serves the interactive dashboard HTML
- **CORS Enabled**: Supports cross-origin requests for local development

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the dashboard HTML |
| `/api/fetch-news` | GET/POST | Fetch fresh news for all countries |
| `/api/news/<country>` | GET | Get cached news for a specific country |
| `/api/news-all` | GET | Get all cached news data |
| `/api/status` | GET | Server status and endpoint list |

#### Usage

```bash
# Start the server
cd 1_cashflow_forecast
python news_server.py

# Server runs on http://localhost:5001
```

**API Examples:**
```bash
# Fetch news for all countries
curl http://localhost:5001/api/fetch-news?max_articles=10

# Get news for a specific country
curl http://localhost:5001/api/news/Thailand

# Get all cached news
curl http://localhost:5001/api/news-all
```

#### Country Coverage

The server fetches news for all 8 APAC entities:
- **TH10** → Thailand
- **TW10** → Taiwan
- **SS10** → Singapore
- **MY10** → Malaysia
- **VN20** → Vietnam
- **KR10** → South Korea
- **ID10** → Indonesia
- **PH10** → Philippines

Sentiment data is cached in `News_scraper/country_sentiments/` directory as JSON files.

---

## 📊 Data Sources

**Location:** `Data/`

| File | Description | Records |
|------|-------------|---------|
| `Data - Main.csv` | Primary transaction data | ~84,000+ |
| `Data - Cash Balance.csv` | Cash balance records | Entity-level |
| `Others - Category Linkage.csv` | Category mappings | Reference |
| `Others - Country Mapping.csv` | Entity to country mapping | 8 entities |
| `Others - Exchange Rate.csv` | FX rates | Historical |

### Transaction Categories

| Category | Type | Description |
|----------|------|-------------|
| AP | Outflow | Accounts Payable |
| AR | Inflow | Accounts Receivable |
| Payroll | Outflow | Employee payments |
| Tax payable | Outflow | Tax obligations |
| Bank charges | Outflow | Banking fees |
| Netting AP/AR | Mixed | Intercompany netting |
| Dividend payout | Outflow | Dividend distributions |

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- pip package manager
- CUDA-capable GPU (optional, for faster sentiment analysis)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd UMDAC

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for news scraping)
playwright install chromium
```

### Dependencies

Key dependencies from `requirements.txt`:

```
# Data Processing
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=2.8.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
statsmodels>=0.14.0
prophet>=1.1.0

# NLP & Sentiment
transformers
torch
newspaper4k
lxml_html_clean

# Web Scraping
playwright
feedparser
beautifulsoup4>=4.12.0
lxml>=5.0.0

# Web Server
flask>=2.3.0
flask-cors>=4.0.0

# Notebook Support (optional)
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
```

---

## 🚀 Usage

### 1. Run Cash Flow Forecasting Pipeline

```bash
cd 1_cashflow_forecast
python run_full_pipeline.py
```

**Outputs:**
- Cleaned data: `processed_data/clean_transactions.csv`
- Weekly features: `processed_data/weekly_entity_features.csv`
- Dashboard: `outputs/dashboards/interactive_dashboard.html`

### 2. Run Anomaly Detection

```bash
cd multi_agent_anomaly_detection

# Detect anomalies in data
python run_detection.py --data ../1_cashflow_forecast/processed_data/weekly_ID10.csv --entity ID10

# Generate HTML report
python run_detection.py --data data.csv --output report.html --format html

# Show system status
python run_detection.py --status
```

### 3. Analyze News Sentiment

```bash
cd News_scraper
python main_scraper.py
```

**Using the API programmatically:**
```python
# Option 1: Import from module (if News_scraper is in path)
from main_scraper import analyze_news_sync, get_full_summary

# Option 2: Import with full path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "News_scraper"))
from main_scraper import analyze_news_sync, get_full_summary

# Analyze news synchronously
results = analyze_news_sync(
    keywords=["US economy", "Federal Reserve"],
    max_articles=10,
    fast_mode=True  # Fast mode skips full extraction for strong headlines
)

# Get summary
summary = get_full_summary(results)
print(f"Verdict: {summary['verdict']}")
print(f"Score: {summary['score']}")
```

### 4. Run News Fetch API Server

```bash
cd 1_cashflow_forecast
python news_server.py
```

The server will start on `http://localhost:5001` and provide API endpoints for fetching news. The dashboard can call these endpoints to update news data dynamically.

### 5. Run Full Anomaly Detection Pipeline

```bash
cd multi_agent_anomaly_detection
python run_full_pipeline.py
```

This runs the complete anomaly detection pipeline with dashboard generation for all entities.

---

## 📁 Output Files

### Generated Reports

| File | Location | Description |
|------|----------|-------------|
| `interactive_dashboard.html` | `1_cashflow_forecast/outputs/dashboards/` | Main cash flow forecast dashboard |
| `anomaly_detection_dashboard.html` | `multi_agent_anomaly_detection/outputs/dashboards/` | Anomaly detection dashboard |
| `detection_summary.json` | `multi_agent_anomaly_detection/outputs/reports/` | Summary of anomaly detections |
| `detailed_verdicts.json` | `multi_agent_anomaly_detection/outputs/reports/` | Detailed verdict explanations |
| `sentiment_report.json` | `News_scraper/` | Latest sentiment analysis results |
| `*_sentiment.json` | `News_scraper/country_sentiments/` | Country-specific sentiment data |

### Processed Data

| File | Location | Description |
|------|----------|-------------|
| `clean_transactions.csv` | `1_cashflow_forecast/processed_data/` | Cleaned transaction data |
| `weekly_entity_features.csv` | `1_cashflow_forecast/processed_data/` | Aggregated weekly features for all entities |
| `weekly_ID10.csv` | `1_cashflow_forecast/processed_data/` | Per-entity weekly data (ID10) |
| `weekly_TW10.csv` | `1_cashflow_forecast/processed_data/` | Per-entity weekly data (TW10) |
| `weekly_*.csv` | `1_cashflow_forecast/processed_data/` | Per-entity weekly data (other entities) |
| `knowledge_base.db` | `multi_agent_anomaly_detection/data/` | SQLite database for anomaly detection knowledge base |

---

## 📂 Project Structure

```
UMDAC/
├── 📁 1_cashflow_forecast/
│   ├── run_full_pipeline.py         # Main ML forecasting script
│   ├── news_server.py               # Flask API server for news fetching
│   ├── PIPELINE_DOCUMENTATION.md    # Technical documentation
│   ├── NEWS_FETCH_README.md         # News API documentation
│   ├── processed_data/              # Cleaned/feature data
│   │   ├── clean_transactions.csv
│   │   ├── weekly_entity_features.csv
│   │   └── weekly_*.csv (per entity)
│   └── outputs/
│       └── dashboards/
│           └── interactive_dashboard.html
│
├── 📁 multi_agent_anomaly_detection/
│   ├── agents/                      # Detection agent implementations
│   │   ├── base_agent.py
│   │   ├── statistical_agent.py
│   │   ├── pattern_agent.py
│   │   ├── rule_agent.py
│   │   ├── temporal_agent.py
│   │   └── category_agent.py
│   ├── coordination/
│   │   └── meta_coordinator.py      # Agent orchestration
│   ├── core/                        # Core components
│   │   ├── models.py
│   │   ├── knowledge_base.py
│   │   ├── rule_graph.py
│   │   └── interpretable_tree.py
│   ├── evolution/                   # Rule evolution & feedback
│   │   ├── rule_evolution.py
│   │   └── feedback.py
│   ├── rules/                       # Rule definitions (JSON)
│   │   ├── business_rules.json
│   │   ├── statistical_rules.json
│   │   └── temporal_rules.json
│   ├── data/
│   │   └── knowledge_base.db        # SQLite database
│   ├── outputs/
│   │   ├── dashboards/
│   │   │   └── anomaly_detection_dashboard.html
│   │   └── reports/
│   │       ├── detection_summary.json
│   │       └── detailed_verdicts.json
│   ├── utils/
│   │   └── helpers.py
│   ├── run_detection.py             # CLI entry point
│   └── run_full_pipeline.py        # End-to-end pipeline
│
├── 📁 News_scraper/
│   ├── main_scraper.py              # Main orchestrator with API
│   ├── news_scraper.py              # RSS/web scraping
│   ├── article_extractor.py         # Content extraction
│   ├── sentiment_analyzer.py        # FinBERT analysis
│   ├── country_sentiments/          # Cached sentiment data
│   │   └── *_sentiment.json (per country)
│   └── sentiment_report.json        # Latest analysis results
│
├── 📁 Data/
│   ├── Datathon Dataset.xlsx - Data - Main.csv
│   ├── Datathon Dataset.xlsx - Data - Cash Balance.csv
│   ├── Datathon Dataset.xlsx - Others - Category Linkage.csv
│   ├── Datathon Dataset.xlsx - Others - Country Mapping.csv
│   └── Datathon Dataset.xlsx - Others - Exchange Rate.csv
│
├── 📁 outputs/
│   └── agent_reports/               # Additional agent reports
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🛠 Technical Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn, XGBoost, LightGBM |
| **NLP** | Transformers (FinBERT), PyTorch |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Plotly.js, Matplotlib, Seaborn |
| **Web Scraping** | Playwright, Newspaper4k, feedparser, BeautifulSoup4 |
| **Web Server** | Flask, Flask-CORS |
| **Database** | SQLite |
| **Dashboard** | HTML5/CSS3/JavaScript (Plotly.js) |

---

## 📈 Performance Metrics

### Forecast Accuracy (Backtest)

| Entity | RMSE (USD) | MAE (USD) | Direction Accuracy |
|--------|------------|-----------|-------------------|
| TW10 | $85,972 | $69,902 | ~60% |
| PH10 | $113,941 | $95,183 | ~58% |
| **TH10** | **$17,447** | **$14,704** | **~65%** |
| ID10 | $67,938 | $59,772 | ~55% |
| SS10 | $2,314 | $2,065 | ~70% |

### Anomaly Detection

- **Agents**: 5 specialized detection agents (Statistical, Pattern, Rule, Temporal, Category)
- **Resolution**: Consensus-based conflict resolution via MetaCoordinator
- **Confidence**: Weighted ensemble voting with confidence scores
- **Evolution**: Automatic rule optimization via feedback mechanism
- **Interpretability**: Detailed explanations for each anomaly detection
- **Knowledge Base**: Persistent SQLite storage for rules and historical detections

---

## 📄 License

This project was developed for the AstraZeneca DATATHON 2025.

---

<p align="center">
  <em>Built with ❤️ for AstraZeneca Treasury Operations</em>
</p>

