import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 設定時間範圍
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

def download_stock_data(ticker, start_date, end_date):
    """下載股票數據"""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def add_technical_indicators(df):
    """添加技術指標"""
    # 計算移動平均線
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    
    # 計算RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # 計算MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # 計算布林帶
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    return df

def generate_signals(df):
    """生成交易信號"""
    # 移動平均線交叉信號
    df['SMA_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
    
    # 價格突破布林帶
    df['BB_Breakout'] = np.where(df['Close'] > df['BBU_20_2.0'], 1, 
                               np.where(df['Close'] < df['BBL_20_2.0'], -1, 0))
    
    # MACD信號
    df['MACD_Signal'] = np.where(df['MACD_12_26_9'] > df['MACDs_12_26_9'], 1, -1)
    
    # RSI信號
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    
    # 綜合信號（加權平均）
    df['Signal'] = (df['SMA_Cross'] * 0.3 + 
                   df['BB_Breakout'] * 0.2 + 
                   df['MACD_Signal'] * 0.3 + 
                   df['RSI_Signal'] * 0.2)
    
    # 生成買入/賣出信號
    df['Buy_Signal'] = df['Signal'] > 0.5
    df['Sell_Signal'] = df['Signal'] < -0.5
    
    return df

def plot_stock_data(df, ticker):
    """繪製股票圖表"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=('股價走勢', '成交量'),
                       row_heights=[0.7, 0.3])
    
    # 繪製K線圖
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='K線'),
                 row=1, col=1)
    
    # 繪製移動平均線
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                            line=dict(color='blue', width=1.5), 
                            name='20日均線'),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                            line=dict(color='red', width=1.5), 
                            name='50日均線'),
                 row=1, col=1)
    
    # 繪製布林帶
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], 
                            line=dict(color='grey', width=1), 
                            name='上軌'),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], 
                            line=dict(color='grey', width=1),
                            fill='tonexty',
                            name='下軌'),
                 row=1, col=1)
    
    # 繪製買入信號
    buy_signals = df[df['Buy_Signal']]
    fig.add_trace(go.Scatter(x=buy_signals.index, 
                            y=buy_signals['Low'] * 0.98,
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='green'),
                            name='買入信號'),
                 row=1, col=1)
    
    # 繪製賣出信號
    sell_signals = df[df['Sell_Signal']]
    fig.add_trace(go.Scatter(x=sell_signals.index, 
                             y=sell_signals['High'] * 1.02,
                             mode='markers',
                             marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='賣出信號'),
                 row=1, col=1)
    
    # 繪製成交量
    colors = ['green' if row['Open'] - row['Close'] >= 0 
              else 'red' for index, row in df.iterrows()]
    
    fig.add_trace(go.Bar(x=df.index, 
                        y=df['Volume'],
                        marker_color=colors,
                        name='成交量'),
                 row=2, col=1)
    
    # 更新佈局
    fig.update_layout(
        title=f'{ticker} 股票分析',
        xaxis_title='日期',
        yaxis_title='價格 (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=800
    )
    
    # 更新y軸標題
    fig.update_yaxes(title_text="價格 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    # 顯示圖表
    fig.show()

def backtest_strategy(df, initial_capital=10000, stop_loss_pct=0.05, take_profit_pct=0.1):
    """回測交易策略"""
    position = 0  # 0: 空倉, 1: 持有多頭, -1: 持有空頭
    entry_price = 0
    portfolio_value = [initial_capital]
    trades = []
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        prev_signal = df['Signal'].iloc[i-1]
        
        # 止損/止盈檢查
        if position == 1:  # 多頭持倉
            if current_price <= entry_price * (1 - stop_loss_pct):
                # 觸發止損
                pct_change = -stop_loss_pct
                position = 0
                trades.append(('SELL', df.index[i], current_price, 'STOP_LOSS'))
            elif current_price >= entry_price * (1 + take_profit_pct):
                # 觸發止盈
                pct_change = take_profit_pct
                position = 0
                trades.append(('SELL', df.index[i], current_price, 'TAKE_PROFIT'))
            else:
                pct_change = (current_price / df['Close'].iloc[i-1]) - 1
        elif position == -1:  # 空頭持倉
            if current_price >= entry_price * (1 + stop_loss_pct):
                # 觸發止損
                pct_change = -stop_loss_pct
                position = 0
                trades.append(('COVER', df.index[i], current_price, 'STOP_LOSS'))
            elif current_price <= entry_price * (1 - take_profit_pct):
                # 觸發止盈
                pct_change = take_profit_pct
                position = 0
                trades.append(('COVER', df.index[i], current_price, 'TAKE_PROFIT'))
            else:
                pct_change = (df['Close'].iloc[i-1] / current_price) - 1
        else:  # 空倉
            pct_change = 0
            
            # 生成交易信號
            if prev_signal > 0.5:  # 買入信號
                position = 1
                entry_price = current_price
                trades.append(('BUY', df.index[i], current_price, 'SIGNAL'))
            elif prev_signal < -0.5:  # 賣空信號
                position = -1
                entry_price = current_price
                trades.append(('SHORT', df.index[i], current_price, 'SIGNAL'))
        
        # 更新投資組合價值
        portfolio_value.append(portfolio_value[-1] * (1 + pct_change * abs(position)))
    
    # 計算績效指標
    returns = pd.Series(portfolio_value).pct_change().dropna()
    total_return = (portfolio_value[-1] / initial_capital - 1) * 100
    annualized_return = (1 + total_return/100) ** (252/len(df)) - 1
    sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    max_drawdown = (pd.Series(portfolio_value) / pd.Series(portfolio_value).cummax() - 1).min() * 100
    
    return {
        'portfolio_value': portfolio_value,
        'trades': trades,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades)
    }

def prepare_prediction_data(df, forecast_days=5):
    """準備預測數據"""
    # 創建特徵
    df_pred = df[['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD_12_26_9', 'MACDs_12_26_9']].copy()
    
    # 創建滯動特徵
    for i in range(1, 6):
        df_pred[f'Close_Lag_{i}'] = df_pred['Close'].shift(i)
        df_pred[f'Volume_Lag_{i}'] = df_pred['Volume'].shift(i)
    
    # 創建目標變量（未來5天收盤價）
    for i in range(1, forecast_days+1):
        df_pred[f'Target_{i}'] = df['Close'].shift(-i)
    
    # 刪除包含NaN的行
    df_pred = df_pred.dropna()
    
    return df_pred

def train_prediction_model(X, y):
    """訓練預測模型"""
    # 標準化特徵
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 只預測下一天的價格（第一個目標變量）
    y_single = y[:, 0]
    
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_single, test_size=0.2, random_state=42
    )
    
    # 訓練隨機森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 評估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2

def predict_future_prices(model, scaler, last_data, forecast_days=5):
    """預測未來價格"""
    predictions = []
    current_features = last_data.copy().reshape(1, -1)
    
    for _ in range(forecast_days):
        # 標準化特徵
        scaled_features = scaler.transform(current_features)
        
        # 預測下一天
        pred = model.predict(scaled_features)[0]
        predictions.append(pred)
        
        # 更新特徵用於下一次預測
        current_features = np.roll(current_features, 1, axis=1)
        current_features[0, 0] = pred
    
    return predictions

def analyze_stock(ticker='TSLA', initial_capital=10000, forecast_days=5):
    """分析股票"""
    print(f"正在下載 {ticker} 的股票數據...")
    df = download_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        print("無法下載股票數據，請檢查股票代碼或網絡連接。")
        return
    
    print("正在計算技術指標...")
    df = add_technical_indicators(df)
    df = generate_signals(df)
    
    # 回測策略
    print("\n正在回測交易策略...")
    backtest_results = backtest_strategy(df, initial_capital=initial_capital)
    
    # 準備預測數據
    print("\n正在準備預測數據...")
    df_pred = prepare_prediction_data(df, forecast_days)
    
    # 分割特徵和目標
    feature_cols = [col for col in df_pred.columns if not col.startswith('Target_')]
    target_cols = [f'Target_{i+1}' for i in range(forecast_days)]
    
    X = df_pred[feature_cols].values
    y = df_pred[target_cols].values
    
    # 訓練模型
    print("正在訓練預測模型...")
    model, scaler, mse, r2 = train_prediction_model(X, y)
    
    # 使用最後的數據進行預測
    last_data = X[-1].copy()
    predictions = predict_future_prices(model, scaler, last_data, forecast_days)
    
    # 顯示最新數據
    latest = df.iloc[-1]
    print("\n=== 最新數據分析 ===")
    print(f"日期: {latest.name.strftime('%Y-%m-%d')}")
    print(f"收盤價: ${latest['Close']:.2f}")
    print(f"20日均線: ${latest['SMA_20']:.2f}")
    print(f"50日均線: ${latest['SMA_50']:.2f}")
    print(f"RSI (14天): {latest['RSI']:.2f}")
    
    # 生成交易建議
    print("\n=== 交易建議 ===")
    
    if latest['Signal'] > 0.5:
        print("✅ 強烈買入信號: 多個技術指標顯示買入機會")
        print(f"   綜合信號強度: {latest['Signal']:.2f}")
    elif latest['Signal'] < -0.5:
        print("❌ 強烈賣出信號: 多個技術指標顯示賣出機會")
        print(f"   綜合信號強度: {latest['Signal']:.2f}")
    else:
        print("➖ 中性信號: 建議觀望")
    
    if latest['RSI'] > 70:
        print(f"⚠️ 超買警報: RSI {latest['RSI']:.2f} (高於70)")
    elif latest['RSI'] < 30:
        print(f"🟢 超賣機會: RSI {latest['RSI']:.2f} (低於30)")
    
    # 顯示價格預測
    print("\n=== 未來價格預測 ===")
    print(f"模型 R² 分數: {r2:.4f}")
    print(f"模型均方誤差: {mse:.4f}")
    print("\n預測未來價格走勢:")
    
    last_date = df.index[-1]
    for i, pred in enumerate(predictions, 1):
        pred_date = last_date + pd.Timedelta(days=i)
        print(f"{pred_date.strftime('%Y-%m-%d')}: 預測收盤價 ${pred:.2f}")
    
    # 計算預期報酬率
    current_price = latest['Close']
    predicted_prices = [current_price] + predictions
    returns = [(predicted_prices[i] - predicted_prices[0]) / predicted_prices[0] * 100 
              for i in range(1, len(predicted_prices))]
    
    print("\n預期報酬率:")
    for i, ret in enumerate(returns, 1):
        print(f"{i}天後: {ret:.2f}%")
    
    # 顯示回測結果
    print("\n=== 策略回測結果 ===")
    print(f"初始資金: ${initial_capital:,.2f}")
    print(f"最終資金: ${backtest_results['portfolio_value'][-1]:,.2f}")
    print(f"總報酬率: {backtest_results['total_return']:.2f}%")
    print(f"年化報酬率: {backtest_results['annualized_return']*100:.2f}%")
    print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {backtest_results['max_drawdown']:.2f}%")
    print(f"交易次數: {backtest_results['num_trades']}")
    
    # 繪製圖表
    print("\n正在生成圖表...")
    plot_stock_data(df, ticker)
    
    # 繪製投資組合價值曲線
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=backtest_results['portfolio_value'],
        mode='lines',
        name='投資組合價值',
        line=dict(color='green')
    ))
    
    # 標記交易點
    if backtest_results['trades']:
        trades_df = pd.DataFrame(backtest_results['trades'], 
                               columns=['type', 'date', 'price', 'reason'])
        
        buy_trades = trades_df[trades_df['type'].isin(['BUY', 'SHORT'])]
        sell_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER'])]
        
        fig.add_trace(go.Scatter(
            x=buy_trades['date'],
            y=[backtest_results['portfolio_value'][df.index.get_loc(d)] for d in buy_trades['date']],
            mode='markers',
            name='買入點',
            marker=dict(symbol='triangle-up', size=10, color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_trades['date'],
            y=[backtest_results['portfolio_value'][df.index.get_loc(d)] for d in sell_trades['date']],
            mode='markers',
            name='賣出點',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
    
    fig.update_layout(
        title='投資組合價值曲線',
        xaxis_title='日期',
        yaxis_title='投資組合價值 ($)',
        template='plotly_white',
        height=600
    )
    fig.show()
    
    # 繪製價格預測圖
    fig = go.Figure()
    
    # 歷史價格
    fig.add_trace(go.Scatter(
        x=df.index[-30:],  # 顯示最近30天的歷史數據
        y=df['Close'][-30:],
        mode='lines',
        name='歷史價格',
        line=dict(color='blue')
    ))
    
    # 預測價格
    future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(forecast_days+1)]
    pred_prices = [df['Close'].iloc[-1]] + predictions
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_prices,
        mode='lines+markers',
        name='預測價格',
        line=dict(color='red', dash='dash')
    ))
    
    # 添加置信區間（簡單起見，這裡使用固定百分比）
    confidence = 0.02  # 2% 的置信區間
    upper_bound = [p * (1 + confidence) for p in pred_prices]
    lower_bound = [p * (1 - confidence) for p in pred_prices]
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='95% 置信區間'
    ))
    
    fig.update_layout(
        title=f'{ticker} 價格預測 ({forecast_days}天)',
        xaxis_title='日期',
        yaxis_title='價格 ($)',
        template='plotly_white',
        height=600
    )
    fig.show()

if __name__ == "__main__":
    analyze_stock('TSLA', forecast_days=25)
