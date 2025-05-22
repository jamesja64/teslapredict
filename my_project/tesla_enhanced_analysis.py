import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# 下載NLTK數據
nltk.download('vader_lexicon')

class EnhancedStockAnalyzer:
    def __init__(self, ticker, start_date=None, end_date=None):
        self.ticker = ticker
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.df = None
        self.scaler = MinMaxScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.setup_twitter_api()
        
        # 初始化金融情感分析模型
        self.finbert = None
        try:
            # 嘗試加載金融情感分析模型
            self.finbert = pipeline(
                "text-classification",
                model="yiyanghkust/finbert-tone",
                tokenizer="yiyanghkust/finbert-tone"
            )
            print("成功加載FinBERT金融情感分析模型")
        except Exception as e:
            print(f"警告: 無法加載FinBERT模型: {e}")
            print("將繼續使用基本情感分析方法")
    
    def setup_twitter_api(self):
        """設置Twitter API憑證"""
        # 注意：使用前請替換為您的API密鑰
        self.twitter_auth = {
            'consumer_key': 'YOUR_CONSUMER_KEY',
            'consumer_secret': 'YOUR_CONSUMER_SECRET',
            'access_token': 'YOUR_ACCESS_TOKEN',
            'access_token_secret': 'YOUR_ACCESS_TOKEN_SECRET'
        }
        
        try:
            self.twitter_client = tweepy.Client(
                bearer_token='YOUR_BEARER_TOKEN',
                consumer_key=self.twitter_auth['consumer_key'],
                consumer_secret=self.twitter_auth['consumer_secret'],
                access_token=self.twitter_auth['access_token'],
                access_token_secret=self.twitter_auth['access_token_secret']
            )
        except Exception as e:
            print(f"Twitter API初始化失敗: {e}")
            self.twitter_client = None
    
    def fetch_stock_data(self):
        """下載股票數據"""
        print(f"正在下載 {self.ticker} 的股票數據...")
        stock = yf.Ticker(self.ticker)
        self.df = stock.history(start=self.start_date, end=self.end_date)
        return self.df
    
    def add_technical_indicators(self):
        """添加技術指標"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        print("正在計算技術指標...")
        df = self.df.copy()
        
        # 移動平均線
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        # 動量指標
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        # 波動率指標
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)
        
        # 成交量指標
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        self.df = df
        return df
    
    def fetch_news_sentiment(self):
        """獲取新聞情緒分析"""
        print("正在分析新聞情緒...")
        try:
            # 從Yahoo Finance獲取新聞
            url = f"https://finance.yahoo.com/quote/{self.ticker}/news?p={self.ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = []
            for item in soup.select('h3'):
                text = item.get_text()
                if text and len(text) > 10:
                    # 使用TextBlob進行基本情感分析
                    blob = TextBlob(text)
                    blob_sentiment = blob.sentiment.polarity
                    
                    # 使用VADER進行情感分析
                    vader_scores = self.sentiment_analyzer.polarity_scores(text)
                    
                    # 使用FinBERT進行金融情感分析（如果可用）
                    finbert_score = 0
                    if self.finbert is not None:
                        try:
                            finbert_result = self.finbert(text[:512])[0]  # 限制文本長度
                            finbert_sentiment = 1.0 if finbert_result['label'] == 'Positive' else -1.0
                            finbert_score = finbert_result['score'] * finbert_sentiment
                        except Exception as e:
                            print(f"FinBERT分析出錯: {e}")
                            finbert_score = 0
                    
                    # 加權平均三種情緒分數
                    sentiment_score = (vader_scores['compound'] * 0.4 + 
                                     blob_sentiment * 0.3 + 
                                     finbert_score * 0.3)
                    
                    news_items.append({
                        'title': text,
                        'vader_score': vader_scores['compound'],
                        'textblob_score': blob_sentiment,
                        'finbert_score': finbert_score,
                        'composite_score': sentiment_score,
                        'source': 'Yahoo Finance'
                    })
            
            # 獲取Twitter情緒
            if self.twitter_client:
                try:
                    tweets = self.twitter_client.search_recent_tweets(
                        f"${self.ticker} -is:retweet",
                        max_results=50,
                        tweet_fields=['created_at', 'public_metrics']
                    )
                    
                    if tweets.data:
                        for tweet in tweets.data:
                            text = tweet.text
                            vader_scores = self.sentiment_analyzer.polarity_scores(text)
                            blob = TextBlob(text)
                            blob_sentiment = blob.sentiment.polarity
                            
                            # 計算加權情緒分數
                            sentiment_score = (vader_scores['compound'] * 0.6 + 
                                             blob_sentiment * 0.4)
                            
                            news_items.append({
                                'title': text,
                                'vader_score': vader_scores['compound'],
                                'textblob_score': blob_sentiment,
                                'finbert_score': 0,  # 跳過FinBERT處理推文以節省時間
                                'composite_score': sentiment_score,
                                'source': 'Twitter',
                                'likes': tweet.public_metrics['like_count'],
                                'retweets': tweet.public_metrics['retweet_count']
                            })
                except Exception as e:
                    print(f"獲取Twitter數據時出錯: {e}")
            
            # 計算平均情緒分數
            if news_items:
                avg_sentiment = sum(item['composite_score'] for item in news_items) / len(news_items)
                return {
                    'sentiment_score': avg_sentiment,
                    'news_count': len(news_items),
                    'news_items': sorted(news_items, key=lambda x: abs(x['composite_score']), reverse=True)[:5]
                }
            
        except Exception as e:
            print(f"獲取新聞時出錯: {e}")
        
        return {
            'sentiment_score': 0,
            'news_count': 0,
            'news_items': []
        }
    
    def prepare_ml_data(self, forecast_days=5):
        """準備機器學習數據"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        print("準備機器學習數據...")
        df = self.df.copy()
        
        # 創建目標變量：未來N天的收盤價
        df['Target'] = df['Close'].shift(-forecast_days)
        
        # 創建特徵
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'SMA_20', 'SMA_50', 'SMA_200', 'RSI',
                   'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                   'ATR', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0',
                   'OBV']
        
        # 添加技術指標的滯動特徵
        for col in ['Close', 'Volume', 'RSI', 'MACD_12_26_9', 'ATR']:
            for window in [5, 10, 20]:
                df[f'{col}_MA_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_STD_{window}'] = df[col].rolling(window=window).std()
                features.extend([f'{col}_MA_{window}', f'{col}_STD_{window}'])
        
        # 添加滯動特徵
        for col in ['Close', 'Volume', 'RSI']:
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
                features.append(f'{col}_lag{lag}')
        
        # 添加技術指標
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=21).std() * np.sqrt(252)  # 年化波動率
        features.extend(['Price_Change', 'Volatility'])
        
        # 刪除包含NaN的行
        df = df.dropna()
        
        # 確保特徵存在
        features = [f for f in features if f in df.columns]
        
        # 準備特徵和目標
        X = df[features]
        y = df['Target']
        
        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割訓練集和測試集
        split = int(0.8 * len(X))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test, X, y, features
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """訓練多個機器學習模型"""
        print("訓練機器學習模型...")
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # 訓練模型
            model.fit(X_train, y_train)
            
            # 預測
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 計算指標
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': y_pred_test
            }
            
            print(f"\n{name} 模型性能:")
            print(f"訓練集 RMSE: {train_rmse:.2f}")
            print(f"測試集 RMSE: {test_rmse:.2f}")
            print(f"訓練集 R²: {train_r2:.4f}")
            print(f"測試集 R²: {test_r2:.4f}")
        
        return results
    
    def predict_future_prices(self, model, X, forecast_days=5):
        """預測未來價格"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        print(f"預測未來 {forecast_days} 天價格...")
        
        # 獲取最新數據
        last_data = X[-1].reshape(1, -1)
        
        # 預測未來價格
        future_prices = []
        for _ in range(forecast_days):
            # 預測下一天價格
            next_price = model.predict(last_data)[0]
            future_prices.append(next_price)
            
            # 更新特徵用於下一個預測
            # 這裡簡化了特徵更新邏輯，實際應用中需要更複雜的處理
            last_data[0][0] = next_price  # 更新Open
            last_data[0][1] = next_price * 1.01  # 假設High比Open高1%
            last_data[0][2] = next_price * 0.99  # 假設Low比Open低1%
            last_data[0][3] = next_price  # 更新Close
            
            # 更新其他特徵（簡化處理）
            for i in range(4, len(last_data[0])):
                last_data[0][i] = last_data[0][i] * 0.99  # 簡單衰減
        
        return future_prices
    
    def plot_analysis(self, sentiment_results, ml_results, forecast_days=5):
        """繪製分析結果"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        df = self.df.copy()
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f"{self.ticker} 價格走勢與預測",
                "技術指標",
                "市場情緒"
            ),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # 1. 價格走勢與預測
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="K線"
            ),
            row=1, col=1
        )
        
        # 添加移動平均線
        for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
            if ma in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ma],
                        name=ma,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # 添加預測價格
        if ml_results and 'RandomForest' in ml_results:
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            future_prices = ml_results['RandomForest'].get('future_prices', [])
            
            if len(future_prices) == forecast_days:
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=future_prices,
                        name='預測價格',
                        mode='lines+markers',
                        line=dict(color='purple', width=2, dash='dot'),
                        marker=dict(size=8, symbol='diamond')
                    ),
                    row=1, col=1
                )
        
        # 2. 技術指標
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # 添加RSI超買超賣線
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 添加MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACDs_12_26_9'],
                name='Signal',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # 3. 市場情緒
        if sentiment_results and 'news_items' in sentiment_results:
            sentiment_dates = []
            sentiment_scores = []
            
            for i, item in enumerate(sentiment_results['news_items']):
                sentiment_dates.append(datetime.now() - timedelta(days=len(sentiment_results['news_items']) - i - 1))
                sentiment_scores.append(item['composite_score'] * 100)  # 轉換為百分比
            
            fig.add_trace(
                go.Bar(
                    x=sentiment_dates,
                    y=sentiment_scores,
                    name='情緒分數',
                    marker_color=['green' if s > 0 else 'red' for s in sentiment_scores],
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # 更新佈局
        fig.update_layout(
            title=f"{self.ticker} 增強分析報告 ({self.start_date} 至 {self.end_date})",
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        # 更新Y軸標題
        fig.update_yaxes(title_text="價格 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD / 情緒分數", row=3, col=1)
        
        fig.show()
    
    def run_analysis(self, forecast_days=5):
        """運行完整分析"""
        # 1. 獲取數據
        self.fetch_stock_data()
        
        # 2. 計算技術指標
        self.add_technical_indicators()
        
        # 3. 分析市場情緒
        sentiment_results = self.fetch_news_sentiment()
        
        # 4. 準備機器學習數據
        X_train, X_test, y_train, y_test, X, y, features = self.prepare_ml_data(forecast_days)
        
        # 5. 訓練模型
        ml_results = self.train_models(X_train, X_test, y_train, y_test)
        
        # 6. 預測未來價格
        best_model = ml_results['RandomForest']['model']
        future_prices = self.predict_future_prices(best_model, X, forecast_days)
        ml_results['RandomForest']['future_prices'] = future_prices
        
        # 7. 顯示分析結果
        print("\n=== 分析完成 ===")
        print(f"股票代碼: {self.ticker}")
        print(f"分析期間: {self.start_date} 至 {self.end_date}")
        print(f"當前價格: ${self.df['Close'].iloc[-1]:.2f}")
        print(f"市場情緒分數: {sentiment_results.get('sentiment_score', 0):.2f}")
        print(f"未來 {forecast_days} 天預測價格: {[f'${p:.2f}' for p in future_prices]}")
        
        # 8. 繪製圖表
        self.plot_analysis(sentiment_results, ml_results, forecast_days)
        
        return {
            'last_price': self.df['Close'].iloc[-1],
            'sentiment': sentiment_results,
            'ml_results': ml_results,
            'future_prices': future_prices
        }

# 使用示例
if __name__ == "__main__":
    # 創建分析器實例
    analyzer = EnhancedStockAnalyzer(
        ticker='TSLA',
        start_date='2024-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # 運行分析
    results = analyzer.run_analysis(forecast_days=5)
