# Cryptocurrency Sentiment Analysis: Predicting Bitcoin Price Movements

A comprehensive research project investigating whether social media sentiment can predict Bitcoin price movements using multi-source data collection, custom NLP models, and comparative analysis with domain-specific pre-trained models.

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Deep Learning](https://img.shields.io/badge/DL-PyTorch-red)
![NLP](https://img.shields.io/badge/NLP-Transformers-green)

## 🎯 Research Question

**Can social media sentiment predict Bitcoin price movements at hourly timeframes?**

This project explores whether sentiment extracted from Reddit, Telegram, YouTube, and news sources can forecast Bitcoin price direction, comparing a custom sentiment encoder against CryptoBERT, a cryptocurrency-specific pre-trained model.

## 📋 Table of Contents

- [Key Findings](#-key-findings)
- [Features](#-features)
- [Technical Architecture](#-technical-architecture)
- [Data Sources](#-data-sources)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Analysis](#-results--analysis)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Academic Value](#-academic-value)
- [Future Work](#-future-work)
- [License](#-license)

## 🔬 Key Findings

### Primary Findings

1. **Sentiment Doesn't Predict Hourly Prices**
   - Both encoders achieved ~52-53% accuracy (baseline: 50.9%)
   - Only 1.5-2.0% improvement over random guessing
   - Sentiment appears to be **reactive** rather than **predictive**

2. **Encoder Sophistication Doesn't Matter**
   - CryptoBERT (domain-specific, pre-trained) only outperformed custom encoder by 0.5%
   - Problem is lack of predictive signal in data, not encoding quality
   - Custom simple models can match expensive pre-trained alternatives

3. **Practical Implications**
   - **For Traders:** Don't rely on hourly social media sentiment for trading decisions
   - **For Researchers:** Negative results are valuable - knowing what doesn't work prevents wasted effort
   - **For Time Horizons:** Sentiment may be more useful at daily/weekly timeframes (not tested here)

### Model Performance

| Model | Custom Encoder | CryptoBERT | Improvement vs Baseline |
|-------|----------------|------------|------------------------|
| Logistic Regression | 52.4% | 52.9% | +1.5% / +2.0% |
| Random Forest | 51.8% | 51.9% | +0.9% / +1.0% |
| Gradient Boosting | 52.1% | 52.3% | +1.2% / +1.4% |
| **Baseline (Random)** | **50.9%** | **50.9%** | **0%** |

## ✨ Features

### Data Collection Pipeline
- **Multi-Source Aggregation**: Reddit, Telegram, YouTube, Google News
- **Price Data Integration**: CryptoCompare API for Bitcoin prices
- **Automated Collection**: Parallel data fetching with error handling
- **Data Persistence**: Saves to JSON for reproducibility

### Dual Sentiment Analysis Approach
- **Custom Encoder**: Simple neural network trained on Twitter sentiment + crypto examples
- **CryptoBERT**: Pre-trained transformer specifically for cryptocurrency text
- **Comparative Analysis**: Head-to-head encoder performance evaluation

### Machine Learning Pipeline
- **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting
- **Feature Engineering**: Sentiment scores, technical indicators, temporal features
- **Proper Validation**: Train/test split with temporal ordering preserved
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrices

### Visualization & Analysis
- **Performance Comparison**: Training vs test accuracy charts
- **Feature Importance**: Identifying key predictive signals
- **Confusion Matrices**: Understanding classification errors
- **Temporal Analysis**: Price movement distribution over time

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Reddit API → Posts & Comments                      │    │
│  │ Telegram API → Channel Messages                    │    │
│  │ YouTube API → Video Comments                       │    │
│  │ Google News → Crypto Headlines                     │    │
│  │ CryptoCompare → Bitcoin Prices (Hourly)            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  SENTIMENT ENCODING                         │
│  ┌─────────────────────┐      ┌─────────────────────┐      │
│  │   Custom Encoder    │      │    CryptoBERT       │      │
│  │  - Simple NN        │      │  - Pre-trained      │      │
│  │  - Twitter trained  │      │  - Crypto-specific  │      │
│  │  - 64.8% accuracy   │      │  - Transformer      │      │
│  └─────────────────────┘      └─────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │ - Sentiment scores (positive, negative, neutral)   │    │
│  │ - Temporal features (hour, day of week)            │    │
│  │ - Technical indicators (price change, volume)      │    │
│  │ - Aggregated statistics (mean, std, max)           │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Logistic Regression (Best: 52.9% test accuracy)    │    │
│  │ Random Forest (52.3% test accuracy)                │    │
│  │ Gradient Boosting (52.3% test accuracy)            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION & VISUALIZATION                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ - Performance metrics across all models            │    │
│  │ - Feature importance analysis                      │    │
│  │ - Confusion matrices                               │    │
│  │ - Training vs test comparison charts              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Data Sources

### 1. Reddit (r/CryptoCurrency, r/Bitcoin)
- **Posts & Comments**: Community discussions and sentiment
- **Collection Method**: PRAW (Python Reddit API Wrapper)
- **Data Points**: ~500-1000 posts/comments per collection
- **Features**: Text, scores, timestamps

### 2. Telegram
- **Channels**: Popular crypto trading/news channels
- **Collection Method**: Telethon API
- **Data Points**: Real-time messages from active channels
- **Features**: Messages, timestamps, reactions

### 3. YouTube
- **Target**: Crypto-related video comments
- **Collection Method**: YouTube Data API v3
- **Data Points**: Comments from trending crypto videos
- **Features**: Comment text, likes, timestamps

### 4. Google News
- **Query**: Bitcoin and cryptocurrency headlines
- **Collection Method**: GoogleNews API
- **Data Points**: Latest news articles
- **Features**: Headlines, descriptions, publication times

### 5. Bitcoin Prices
- **Source**: CryptoCompare API
- **Granularity**: Hourly OHLCV data
- **History**: 2000+ hours of price data
- **Features**: Open, High, Low, Close, Volume

## 🔬 Methodology

### Phase 1: Data Collection

```python
class PriceDataCollector:
    """Collect Bitcoin price data from CryptoCompare API"""
    
    def get_historical_prices(self, symbol='BTC', limit=2000):
        """Fetch hourly price data"""
        response = requests.get(
            f"{self.base_url}/histohour",
            params={'fsym': symbol, 'tsym': 'USD', 'limit': limit}
        )
        return self._process_price_data(response.json())
```

All data sources collected in parallel with error handling and rate limiting.

### Phase 2: Sentiment Encoding

#### Custom Encoder
```python
# Simple neural network trained on Twitter sentiment dataset
# Augmented with cryptocurrency-specific examples
# Architecture: Embedding → LSTM → Dense
# Accuracy: 64.8% on test set
```

#### CryptoBERT
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Pre-trained model specifically for cryptocurrency text
model = AutoModelForSequenceClassification.from_pretrained(
    "ElKulako/cryptobert"
)
tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")

# Classifies text as Bullish, Bearish, or Neutral
```

### Phase 3: Feature Engineering

Generated features:
- **Sentiment Scores**: Positive, negative, neutral probabilities
- **Aggregated Metrics**: Mean, std, max sentiment over time windows
- **Temporal Features**: Hour of day, day of week
- **Price Features**: Previous price changes, volatility
- **Volume Features**: Trading volume indicators

### Phase 4: Model Training

Three classification algorithms trained on identical features:

1. **Logistic Regression**
   - Linear classifier
   - Fast training and prediction
   - Good baseline model

2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Feature importance analysis

3. **Gradient Boosting**
   - Iterative ensemble method
   - Often achieves best performance
   - Resistant to overfitting

**Target Variable**: Binary classification (price up/down in next hour)

### Phase 5: Evaluation

Metrics tracked:
- **Accuracy**: Overall correct predictions
- **Precision**: True positive rate
- **Recall**: Coverage of positive cases
- **F1-Score**: Harmonic mean of precision/recall
- **Confusion Matrix**: Classification error analysis

## 📥 Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster CryptoBERT inference)

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/crypto-sentiment-analysis.git
cd crypto-sentiment-analysis
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
```
pandas
numpy
scikit-learn
torch
transformers
praw
telethon
google-api-python-client
matplotlib
seaborn
requests
```

3. **Configure API Keys**

Create a `.env` file with your API credentials:
```env
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Telegram API
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash

# YouTube API
YOUTUBE_API_KEY=your_youtube_key
```

## 🚀 Usage

### Running the Complete Pipeline

```bash
jupyter notebook crypto_sentiment_analysis_final.ipynb
```

Or run cells sequentially:

### 1. Data Collection
```python
# Collect from all sources
price_collector = PriceDataCollector()
reddit_collector = RedditDataCollector()
telegram_collector = TelegramDataCollector()
news_collector = NewsDataCollector()

# Fetch data
prices = price_collector.get_historical_prices()
reddit_data = reddit_collector.collect_posts()
telegram_data = telegram_collector.collect_messages()
news_data = news_collector.collect_headlines()
```

### 2. Sentiment Analysis
```python
# Custom encoder
custom_sentiments = encode_with_custom_model(all_text)

# CryptoBERT
cryptobert_sentiments = encode_with_cryptobert(all_text)
```

### 3. Model Training
```python
# Train models with both encoders
results = {}
for encoder_name, features in [('Custom', custom_features), 
                                ('CryptoBERT', cryptobert_features)]:
    results[encoder_name] = train_and_evaluate(features, labels)
```

### 4. Analysis
```python
# Compare results
plot_performance_comparison(results)
analyze_feature_importance(best_model)
generate_confusion_matrices(predictions, actuals)
```

## 📈 Results & Analysis

### Model Performance Comparison

**Best Models (Logistic Regression):**
- **Custom Encoder**: 52.4% test accuracy
- **CryptoBERT**: 52.9% test accuracy
- **Baseline**: 50.9% (random guessing)

### Key Insights

#### 1. Why Sentiment Fails at Hourly Prediction

**Causality Direction:**
```
Market Movement → Sentiment (Reactive)
NOT
Sentiment → Market Movement (Predictive)
```

When Bitcoin price moves, people react on social media. The sentiment is a **lagging indicator**, not a leading one.

#### 2. Time Horizon Matters

Hourly timeframes are too short for sentiment to propagate into price action. Better results might be achieved at:
- **Daily timeframes**: Sentiment can accumulate
- **Weekly timeframes**: Trends become clearer
- **Event-driven**: Major news/announcements

#### 3. Model Complexity Paradox

CryptoBERT's minimal advantage over custom encoder suggests:
- **Diminishing returns** on sophisticated encoding
- **Data quality** matters more than model sophistication
- **Simple models** can be equally effective for this task

### Feature Importance Analysis

Top features from best models:
1. **Previous price change** (most important)
2. **Time of day** (market hours matter)
3. **Volume indicators** (trading activity)
4. Sentiment scores (minimal importance)

This confirms **price history** is more predictive than sentiment.

### Confusion Matrix Interpretation

```
                Predicted Down    Predicted Up
Actual Down         ~50%            ~50%
Actual Up           ~50%            ~50%
```

Nearly random distribution confirms lack of predictive power.

## 🛠️ Technologies Used

### Data Collection
- **PRAW**: Reddit API wrapper
- **Telethon**: Telegram client library
- **google-api-python-client**: YouTube Data API
- **GoogleNews**: News aggregation
- **requests**: HTTP for CryptoCompare API

### NLP & Machine Learning
- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-trained models
- **scikit-learn**: Classical ML algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts (optional)

### Development
- **Jupyter Notebook**: Interactive development
- **Google Colab**: Cloud execution environment

## 📁 Project Structure

```
crypto-sentiment-analysis/
├── crypto_sentiment_analysis_final.ipynb   # Main notebook
├── requirements.txt                         # Dependencies
├── README.md                               # This file
├── data/                                   # Collected data (gitignored)
│   ├── prices.json
│   ├── reddit_posts.json
│   ├── telegram_messages.json
│   ├── youtube_comments.json
│   └── news_headlines.json
├── models/                                 # Saved models
│   ├── custom_encoder.pt
│   └── trained_classifiers.pkl
└── results/                                # Analysis outputs
    ├── performance_charts.png
    ├── confusion_matrices.png
    └── feature_importance.csv
```

## 🎓 Academic Value

### Why Negative Results Matter

This project demonstrates:

1. **Scientific Rigor**: Testing hypotheses even when results are negative
2. **Practical Value**: Saving others from pursuing ineffective strategies
3. **Methodological Soundness**: Proper comparison of approaches
4. **Reproducibility**: Clear methodology and code for verification

### Publication Potential

Suitable for:
- **Conference Papers**: FinTech, Data Science conferences
- **Academic Journals**: Computational Finance journals
- **Technical Reports**: Industry whitepapers
- **Educational Material**: Case study for data science courses

### Learning Outcomes

Skills demonstrated:
- Multi-source data collection and aggregation
- NLP with pre-trained and custom models
- Machine learning pipeline development
- Statistical analysis and interpretation
- Critical thinking about model limitations

## 🔮 Future Work

### Potential Improvements

1. **Longer Timeframes**
   - Test daily/weekly predictions
   - Compare across multiple time horizons
   - Analyze sentiment momentum

2. **Advanced Features**
   - Social media influence scores
   - Sentiment change velocity
   - Cross-platform sentiment correlation
   - Technical analysis integration

3. **Better Models**
   - LSTM for sequential dependencies
   - Attention mechanisms for key events
   - Ensemble methods combining multiple signals
   - Deep learning architectures

4. **More Data Sources**
   - Twitter/X real-time stream
   - Discord communities
   - On-chain metrics (transaction volume, whale movements)
   - Traditional financial news (Bloomberg, Reuters)

5. **Market Context**
   - Separate bull/bear market analysis
   - Event-driven prediction (halving, regulatory news)
   - Multi-cryptocurrency comparison
   - Market regime classification

### Research Questions

- Does sentiment predict altcoin prices better than Bitcoin?
- How does sentiment lag vary across different time horizons?
- Can sentiment identify market turning points?
- Does sentiment work better for sudden crashes vs rallies?

## 🤝 Contributing

Contributions welcome! Areas for contribution:

- Additional data sources
- Improved feature engineering
- Alternative modeling approaches
- Extended analysis periods
- Cross-validation strategies
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **CryptoBERT**: ElKulako for the pre-trained cryptocurrency model
- **HuggingFace**: Transformers library and model hub
- **CryptoCompare**: Free API for price data
- **Reddit/Telegram/YouTube**: Platform APIs for data access


## 📚 References

1. Crypto sentiment analysis literature
2. Time series prediction papers
3. Social media analytics research
4. FinTech machine learning applications

---

**Note**: This project is for educational and research purposes only. Not financial advice. Cryptocurrency trading involves substantial risk.

**Made with 🧠 for academic research and learning**
