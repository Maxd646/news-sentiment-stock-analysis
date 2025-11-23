# Predicting Price Moves with News Sentiment — Week 1 Challenge

This project explores the impact of financial news on stock price movements, aiming to provide actionable insights for Nova Financial Solutions. The analysis combines **quantitative stock data** with **qualitative news sentiment**, integrating Data Engineering (DE), Financial Analytics (FA), and Machine Learning Engineering (MLE) skills.

## Challenge Overview
The goal is to analyze a large corpus of financial news and stock prices to uncover correlations between news sentiment and market movements. The project simulates real-world financial analytics challenges, emphasizing adaptability, innovative thinking, and technical proficiency.

## Business Objective
- Quantify sentiment from financial news headlines using NLP techniques.
- Establish correlations between sentiment and stock price movements.
- Provide actionable insights and predictive strategies for investment decisions.

## Dataset Overview
**Financial News and Stock Price Integration Dataset (FNSPID):**
- `headline`: News title containing financial events
- `url`: Link to the full article
- `publisher`: Source of the news
- `date`: Publication date and time (UTC-4)
- `stock`: Stock ticker symbol

**Stock Price Data:**
- Open, High, Low, Close (OHLC)
- Volume and daily returns

## Tasks Completed
**Task 1 — Git & Environment Setup:**
- Initialize repository and branches
- Configure Python environment
- Exploratory Data Analysis (EDA) of headlines, publishers, and publishing times

**Task 2 — Technical Indicators & Market Analysis:**
- Clean and prepare stock price data
- Compute technical indicators (SMA, EMA, RSI, MACD) using TA-Lib and PyNance
- Visualize trends and volume patterns

**Task 3 — Sentiment vs Stock Movement Correlation:**
- Generate sentiment scores from news headlines
- Compute daily stock returns
- Perform correlation analysis between sentiment and returns
- Visualize patterns and regression analysis

## Learning Objectives
- Set up a reproducible Python data-science environment with GitHub integration.
- Perform EDA on text and time-series data.
- Calculate financial technical indicators.
- Conduct sentiment analysis using NLP tools.
- Measure correlations between sentiment and stock returns.
- Document findings and create actionable insights.


## Technologies Used
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- TA-Lib, PyNance
- NLTK, TextBlob
- Jupyter Notebook
- Git & GitHub Actions



## References
- [Investopedia: Stock Market](https://www.investopedia.com/terms/s/stockmarket.asp)
- [Investopedia: Stock Analysis](https://www.investopedia.com/terms/s/stock-analysis.asp)
- [Python TextBlob](https://textblob.readthedocs.io/en/dev/)
- [PyNance](https://github.com/mqandil/pynance)
- [TA-Lib Python](https://github.com/ta-lib/ta-lib-python)
- [Git Branching](https://learngitbranching.js.org/)
- [CI/CD Concepts](https://www.atlassian.com/continuous-delivery/ci-vs-ci-vs-cd)
