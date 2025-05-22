import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# 下載NLTK數據
nltk.download('vader_lexicon')

class AdvancedStockAnalyzer:
    def __init__(self, ticker, start_date=None, end_date=None):
        self.ticker = ticker
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.df = None
        self.scaler = StandardScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def fetch_data(self):
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
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # 波動率指標
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_upper'] = bbands['BBU_20_2.0']
        df['BB_middle'] = bbands['BBM_20_2.0']
        df['BB_lower'] = bbands['BBL_20_2.0']
        
        # 成交量指標
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # 動能指標
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_%K'] = stoch['STOCHk_14_3_3']
        df['Stoch_%D'] = stoch['STOCHd_14_3_3']
        
        self.df = df
        return df
    
    def fetch_news_sentiment(self):
        """獲取新聞情緒分析"""
        print("正在分析新聞情緒...")
        try:
            # 這裡使用Yahoo Finance的新聞API
            url = f"https://finance.yahoo.com/quote/{self.ticker}/news?p={self.ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取新聞標題和摘要
            news_items = []
            for item in soup.select('h3'):
                text = item.get_text()
                if text and len(text) > 10:  # 過濾掉太短的文本
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    news_items.append({
                        'title': text,
                        'sentiment': sentiment['compound']
                    })
            
            # 計算平均情緒分數
            if news_items:
                avg_sentiment = sum(item['sentiment'] for item in news_items) / len(news_items)
                return {
                    'sentiment_score': avg_sentiment,
                    'news_count': len(news_items),
                    'news_items': news_items[:5]  # 返回前5條新聞
                }
            
        except Exception as e:
            print(f"獲取新聞時出錯: {e}")
        
        return {
            'sentiment_score': 0,
            'news_count': 0,
            'news_items': []
        }
    
    def analyze_market_regime(self, window=20):
        """分析市場狀態（趨勢/震盪）"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        df = self.df.copy()
        
        # 計算價格變動
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        # 使用ADX判斷趨勢強度
        df['Trend_Strength'] = df['ADX']
        
        # 判斷市場狀態
        df['Market_Regime'] = np.where(
            df['Trend_Strength'] > 25,  # ADX > 25 認為是趨勢市場
            'Trending',
            'Ranging'
        )
        
        return df[['Returns', 'Volatility', 'Trend_Strength', 'Market_Regime']]
    
    def calculate_position_sizing(self, risk_per_trade=0.01, atr_multiplier=2):
        """計算頭寸規模"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        df = self.df.copy()
        
        # 使用ATR計算波動性調整的頭寸規模
        df['Position_Size'] = (risk_per_trade * df['ATR'] * atr_multiplier) / df['Close']
        
        return df[['ATR', 'Position_Size']]
    
    def generate_trading_signals(self):
        """生成交易信號"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        df = self.df.copy()
        
        # 初始化信號列
        df['Signal'] = 0  # 0: 持有, 1: 買入, -1: 賣出
        
        # 基於多個指標生成信號
        # 1. 移動平均線交叉
        df['SMA_Crossover'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        
        # 2. RSI 超買超賣
        df['RSI_Signal'] = 0
        df.loc[df['RSI'] < 30, 'RSI_Signal'] = 1  # 超賣，買入信號
        df.loc[df['RSI'] > 70, 'RSI_Signal'] = -1  # 超買，賣出信號
        
        # 3. MACD 信號
        df['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        
        # 綜合信號
        df['Composite_Signal'] = (
            df['SMA_Crossover'] * 0.4 + 
            df['RSI_Signal'] * 0.3 + 
            df['MACD_Signal'] * 0.3
        )
        
        # 生成最終信號
        df.loc[df['Composite_Signal'] > 0.5, 'Signal'] = 1  # 強烈買入
        df.loc[df['Composite_Signal'] < -0.5, 'Signal'] = -1  # 強烈賣出
        
        self.df = df
        return df
    
    def plot_analysis(self):
        """繪製分析圖表"""
        if self.df is None:
            raise ValueError("請先下載數據")
            
        df = self.df.copy()
        
        # 創建子圖
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{self.ticker} 價格走勢",
                "技術指標",
                "交易信號",
                "成交量"
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. 價格走勢
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
        
        # 3. 交易信號
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='買入信號',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='賣出信號',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
        
        # 4. 成交量
        colors = ['green' if close >= open_ else 'red' 
                 for close, open_ in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='成交量',
                marker_color=colors,
                opacity=0.5
            ),
            row=4, col=1
        )
        
        # 更新佈局
        fig.update_layout(
            title=f"{self.ticker} 高級技術分析 ({self.start_date} 至 {self.end_date})",
            height=1200,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.show()
    
    def run_analysis(self):
        """運行完整分析"""
        # 1. 獲取數據
        self.fetch_data()
        
        # 2. 計算技術指標
        self.add_technical_indicators()
        
        # 3. 生成交易信號
        self.generate_trading_signals()
        
        # 4. 獲取市場情緒
        sentiment = self.fetch_news_sentiment()
        
        # 5. 分析市場狀態
        market_regime = self.analyze_market_regime()
        
        # 6. 計算頭寸規模
        position_sizing = self.calculate_position_sizing()
        
        # 7. 顯示分析結果
        print("\n=== 分析完成 ===")
        print(f"股票代碼: {self.ticker}")
        print(f"分析期間: {self.start_date} 至 {self.end_date}")
        print(f"收盤價: ${self.df['Close'].iloc[-1]:.2f}")
        print(f"市場情緒分數: {sentiment['sentiment_score']:.2f}")
        print(f"當前市場狀態: {market_regime['Market_Regime'].iloc[-1]}")
        print(f"建議頭寸規模: {position_sizing['Position_Size'].iloc[-1]:.4f} 股/每1000美元")
        
        # 8. 繪製圖表
        self.plot_analysis()
        
        return {
            'last_price': self.df['Close'].iloc[-1],
            'sentiment': sentiment,
            'market_regime': market_regime.iloc[-1].to_dict(),
            'position_sizing': position_sizing.iloc[-1].to_dict()
        }

# 使用示例
if __name__ == "__main__":
    # 創建分析器實例
    analyzer = AdvancedStockAnalyzer(
        ticker='TSLA',
        start_date='2023-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # 運行分析
    results = analyzer.run_analysis()
