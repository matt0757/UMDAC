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
  - [Dashboard Enhancer](#4-dashboard-enhancer)
- [Data Sources](#-data-sources)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
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
│   ┌─────────────────┐    ┌─────────────────────┐                ▼                │
│   │  News Sources   │───▶│  News Scraper &     │───▶│ Dashboard Enhancer  │      │
│   │  (RSS/Web)      │    │  Sentiment Analysis │    │ (Market Insights)   │      │
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
| `PIPELINE_DOCUMENTATION.md` | Comprehensive technical documentation |
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
└── run_detection.py         # Main entry point
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

```bash
# Run detection on data
python run_detection.py --data weekly_features.csv --entity ID10

# Train agents on historical data
python run_detection.py --train --data historical_data.csv

# Check system status
python run_detection.py --status

# Run rule evolution cycle
python run_detection.py --evolve
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

### 4. Dashboard Enhancer

**Location:** `dashboard_enhancer.py`

Integrates news sentiment analysis with cash flow forecasts to create enhanced dashboards with market insights.

#### Features

- **Entity-Region Mapping**: Links entities to countries and economic contexts
- **Trend Analysis**: Analyzes forecast trends (direction, volatility)
- **Risk Assessment**: Combined risk scoring from forecast + sentiment
- **Strategic Recommendations**: AI-generated treasury management advice

#### Generated Recommendations

| Outlook | Risk Level | Recommendation |
|---------|------------|----------------|
| **FAVORABLE** | Low-Moderate | Growth initiatives, investment acceleration |
| **CHALLENGING** | Elevated | Conservative stance, increased liquidity |
| **CAUTIOUS** | Moderate-High | Close monitoring, contingency planning |
| **NEUTRAL** | Moderate | Balanced approach, standard operations |

#### Usage

```python
from dashboard_enhancer import run

# Enhances dashboard with Market Insights section
run()
```

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

```
# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=2.8.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
statsmodels>=0.14.0

# NLP & Sentiment
transformers
torch
newspaper4k
lxml_html_clean

# Web Scraping
playwright
feedparser

# Notebook Support (optional)
jupyter>=1.0.0
notebook>=7.0.0
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

### 4. Enhance Dashboard with Market Insights

```bash
python dashboard_enhancer.py
```

---

## 📁 Output Files

### Generated Reports

| File | Location | Description |
|------|----------|-------------|
| `interactive_dashboard.html` | `1_cashflow_forecast/outputs/dashboards/` | Main forecast dashboard |
| `report.html` | Root | Anomaly detection report |
| `sentiment_report.json` | `News_scraper/` | Latest sentiment analysis |

### Processed Data

| File | Location | Description |
|------|----------|-------------|
| `clean_transactions.csv` | `1_cashflow_forecast/processed_data/` | Cleaned transaction data |
| `weekly_entity_features.csv` | `1_cashflow_forecast/processed_data/` | Aggregated features |
| `weekly_*.csv` | `1_cashflow_forecast/processed_data/` | Per-entity weekly data |
| `knowledge_base.db` | `multi_agent_anomaly_detection/data/` | Anomaly detection DB |

---

## 📂 Project Structure

```
UMDAC/
├── 📁 1_cashflow_forecast/
│   ├── run_full_pipeline.py      # Main ML forecasting script
│   ├── PIPELINE_DOCUMENTATION.md # Technical documentation
│   ├── processed_data/           # Cleaned/feature data
│   └── outputs/dashboards/       # Generated dashboards
│
├── 📁 multi_agent_anomaly_detection/
│   ├── agents/                   # Detection agent implementations
│   ├── coordination/             # Meta-coordinator
│   ├── core/                     # Data models, knowledge base
│   ├── evolution/                # Rule evolution, feedback
│   ├── rules/                    # Rule definitions (JSON)
│   ├── data/                     # SQLite database
│   └── run_detection.py          # Entry point
│
├── 📁 News_scraper/
│   ├── main_scraper.py          # Main orchestrator
│   ├── news_scraper.py          # RSS/web scraping
│   ├── article_extractor.py     # Content extraction
│   └── sentiment_analyzer.py    # FinBERT analysis
│
├── 📁 Data/
│   ├── Data - Main.csv          # Transaction data
│   ├── Data - Cash Balance.csv  # Cash positions
│   └── Others - *.csv           # Reference data
│
├── dashboard_enhancer.py        # Sentiment integration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
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
| **Web Scraping** | Playwright, Newspaper4k, feedparser |
| **Database** | SQLite |
| **Dashboard** | HTML5/CSS3/JavaScript |

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

- **Agents**: 5 specialized detection agents
- **Resolution**: Consensus-based conflict resolution
- **Confidence**: Weighted ensemble voting
- **Evolution**: Automatic rule optimization via feedback

---

## 📄 License

This project was developed for the AstraZeneca DATATHON 2025.

---

<p align="center">
  <em>Built with ❤️ for AstraZeneca Treasury Operations</em>
</p>

