
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Minervini SEPA® Trading Assistant",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class MinerviniAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data = None
        self.info = None

    def fetch_data(self, period="2y"):
        """Fetch stock data and info"""
        try:
            self.data = self.stock.history(period=period)
            self.info = self.stock.info
            return True
        except Exception as e:
            st.error(f"Error fetching data for {self.ticker}: {str(e)}")
            return False

    def calculate_moving_averages(self):
        """Calculate key moving averages"""
        if self.data is None or len(self.data) == 0:
            return None

        df = self.data.copy()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_150'] = df['Close'].rolling(window=150).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()

        return df

    def check_trend_template(self):
        """Check Minervini's Trend Template criteria"""
        df = self.calculate_moving_averages()
        if df is None or len(df) < 200:
            return None

        latest = df.iloc[-1]
        current_price = latest['Close']
        ma_50 = latest['MA_50']
        ma_150 = latest['MA_150'] 
        ma_200 = latest['MA_200']

        # Calculate 52-week high/low
        year_ago = df.iloc[-252:] if len(df) >= 252 else df
        week_52_high = year_ago['High'].max()
        week_52_low = year_ago['Low'].min()

        # Check 200-day MA trend (last 30 days)
        ma_200_month_ago = df['MA_200'].iloc[-30] if len(df) >= 30 else ma_200
        ma_200_trending_up = ma_200 > ma_200_month_ago

        # Trend template criteria
        criteria = {
            "Above 150-day MA": current_price > ma_150,
            "Above 200-day MA": current_price > ma_200,
            "150-day above 200-day MA": ma_150 > ma_200,
            "200-day MA trending up": ma_200_trending_up,
            "50-day above 150-day MA": ma_50 > ma_150,
            "50-day above 200-day MA": ma_50 > ma_200,
            "25% above 52-week low": current_price >= week_52_low * 1.25,
            "Within 25% of 52-week high": current_price >= week_52_high * 0.75,
            "Above 50-day MA": current_price > ma_50
        }

        # Calculate distances from MAs
        distances = {
            "Distance from 50-day MA": ((current_price - ma_50) / ma_50 * 100),
            "Distance from 150-day MA": ((current_price - ma_150) / ma_150 * 100),
            "Distance from 200-day MA": ((current_price - ma_200) / ma_200 * 100),
            "Distance from 52-week high": ((current_price - week_52_high) / week_52_high * 100),
            "Distance from 52-week low": ((current_price - week_52_low) / week_52_low * 100)
        }

        return {
            "criteria": criteria,
            "distances": distances,
            "current_price": current_price,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "ma_values": {"MA_50": ma_50, "MA_150": ma_150, "MA_200": ma_200}
        }

    def detect_vcp_pattern(self, lookback=50):
        """Detect Volatility Contraction Pattern"""
        df = self.calculate_moving_averages()
        if df is None or len(df) < lookback:
            return None

        recent_data = df.iloc[-lookback:]

        # Find pivot highs and lows
        highs = []
        lows = []

        for i in range(2, len(recent_data)-2):
            # Pivot high
            if (recent_data.iloc[i]['High'] > recent_data.iloc[i-1]['High'] and 
                recent_data.iloc[i]['High'] > recent_data.iloc[i-2]['High'] and
                recent_data.iloc[i]['High'] > recent_data.iloc[i+1]['High'] and
                recent_data.iloc[i]['High'] > recent_data.iloc[i+2]['High']):
                highs.append((i, recent_data.iloc[i]['High']))

            # Pivot low
            if (recent_data.iloc[i]['Low'] < recent_data.iloc[i-1]['Low'] and 
                recent_data.iloc[i]['Low'] < recent_data.iloc[i-2]['Low'] and
                recent_data.iloc[i]['Low'] < recent_data.iloc[i+1]['Low'] and
                recent_data.iloc[i]['Low'] < recent_data.iloc[i+2]['Low']):
                lows.append((i, recent_data.iloc[i]['Low']))

        # Analyze contraction if we have enough pivots
        if len(highs) >= 3 and len(lows) >= 2:
            # Check for decreasing volatility
            contractions = []
            for i in range(1, min(len(highs), len(lows))):
                if i < len(highs) and i < len(lows):
                    high_idx, high_val = highs[i]
                    low_idx, low_val = lows[i]
                    prev_high_idx, prev_high_val = highs[i-1]
                    prev_low_idx, prev_low_val = lows[i-1]

                    current_range = (high_val - low_val) / low_val * 100
                    prev_range = (prev_high_val - prev_low_val) / prev_low_val * 100

                    contractions.append({
                        'contraction': prev_range - current_range,
                        'range_percent': current_range
                    })

            is_vcp = len([c for c in contractions if c['contraction'] > 0]) >= len(contractions) * 0.6

            return {
                "is_vcp": is_vcp,
                "contractions": contractions,
                "pivot_highs": len(highs),
                "pivot_lows": len(lows)
            }

        return {"is_vcp": False, "contractions": [], "pivot_highs": 0, "pivot_lows": 0}

    def analyze_volume_pattern(self):
        """Analyze volume patterns"""
        if self.data is None or len(self.data) < 50:
            return None

        df = self.data.copy()
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()

        recent_volume = df['Volume'].iloc[-10:].mean()
        avg_volume = df['Volume_MA_50'].iloc[-1]

        return {
            "recent_avg_volume": recent_volume,
            "50_day_avg_volume": avg_volume,
            "volume_ratio": recent_volume / avg_volume if avg_volume > 0 else 0
        }

    def calculate_risk_reward(self, entry_price, stop_loss_percent=7):
        """Calculate risk/reward based on Minervini principles"""
        if self.data is None:
            return None

        # Calculate stop loss
        stop_loss = entry_price * (1 - stop_loss_percent / 100)
        risk_per_share = entry_price - stop_loss
        risk_percent = stop_loss_percent

        # Potential targets based on historical moves
        df = self.calculate_moving_averages()
        recent_high = df['High'].iloc[-50:].max()

        # Conservative target: 2R (2x risk)
        target_2r = entry_price + (2 * risk_per_share)

        # Aggressive target: 3R
        target_3r = entry_price + (3 * risk_per_share)

        # Historical resistance
        target_resistance = recent_high

        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_per_share": risk_per_share,
            "risk_percent": risk_percent,
            "target_2r": target_2r,
            "target_3r": target_3r,
            "target_resistance": target_resistance,
            "reward_risk_2r": 2.0,
            "reward_risk_3r": 3.0
        }

    def calculate_position_size(self, account_size, risk_percent=2, entry_price=None, stop_loss_percent=7):
        """Calculate position size based on risk management"""
        if entry_price is None:
            entry_price = self.data['Close'].iloc[-1]

        # Maximum risk per trade (1.25-2.5% of account per Minervini)
        max_risk_amount = account_size * (risk_percent / 100)

        # Risk per share
        stop_loss = entry_price * (1 - stop_loss_percent / 100)
        risk_per_share = entry_price - stop_loss

        # Position size
        shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * entry_price
        position_percent = (position_value / account_size * 100) if account_size > 0 else 0

        return {
            "shares": shares,
            "position_value": position_value,
            "position_percent": position_percent,
            "risk_amount": shares * risk_per_share,
            "max_recommended_position": min(position_percent, 25)  # Max 25% per Minervini
        }

def create_chart(analyzer):
    """Create comprehensive chart with indicators"""
    df = analyzer.calculate_moving_averages()
    if df is None:
        return None

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price & Moving Averages', 'Volume'),
        vertical_spacing=0.1,
        row_width=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name='50-day MA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_150'], name='150-day MA', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], name='200-day MA', line=dict(color='purple', width=2)), row=1, col=1)

    # Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title=f'{analyzer.ticker} - Technical Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig

def main():
    st.title("📈 Minervini SEPA® Trading Assistant")
    st.markdown("*Based on Mark Minervini's Specific Entry Point Analysis methodology*")

    # Sidebar for inputs
    st.sidebar.header("Trade Setup")

    # Ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()

    # Account parameters
    st.sidebar.subheader("Account Parameters")
    account_size = st.sidebar.number_input("Account Size ($):", min_value=1000, value=100000, step=1000)
    risk_percent = st.sidebar.slider("Risk per Trade (%):", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    stop_loss_percent = st.sidebar.slider("Stop Loss (%):", min_value=3.0, max_value=10.0, value=7.0, step=0.5)

    if st.sidebar.button("Analyze Stock", type="primary"):
        if ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                analyzer = MinerviniAnalyzer(ticker)

                if analyzer.fetch_data():
                    # Main analysis tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trend Analysis", "🔍 Pattern Analysis", "⚖️ Risk/Reward", "💰 Position Sizing"])

                    with tab1:
                        # Stock overview
                        col1, col2, col3, col4 = st.columns(4)

                        current_price = analyzer.data['Close'].iloc[-1]
                        prev_close = analyzer.data['Close'].iloc[-2]
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100

                        col1.metric("Current Price", f"${current_price:.2f}", f"${change:.2f} ({change_pct:+.2f}%)")

                        if analyzer.info:
                            col2.metric("Market Cap", f"${analyzer.info.get('marketCap', 0):,.0f}")
                            col3.metric("Volume", f"{analyzer.data['Volume'].iloc[-1]:,.0f}")
                            col4.metric("Avg Volume (50d)", f"{analyzer.data['Volume'].rolling(50).mean().iloc[-1]:,.0f}")

                        # Chart
                        chart = create_chart(analyzer)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)

                    with tab2:
                        st.subheader("🎯 Minervini Trend Template Analysis")

                        trend_analysis = analyzer.check_trend_template()
                        if trend_analysis:
                            criteria = trend_analysis['criteria']
                            distances = trend_analysis['distances']

                            # Criteria checklist
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Trend Template Criteria:**")
                                passed_criteria = 0
                                for criterion, passed in criteria.items():
                                    icon = "✅" if passed else "❌"
                                    st.markdown(f"{icon} {criterion}")
                                    if passed:
                                        passed_criteria += 1

                                # Overall assessment
                                total_criteria = len(criteria)
                                pass_rate = passed_criteria / total_criteria

                                if pass_rate >= 0.8:
                                    st.markdown('<div class="success-card"><h4>🟢 Strong Stage 2 Uptrend</h4><p>Stock meets most trend template criteria</p></div>', unsafe_allow_html=True)
                                elif pass_rate >= 0.6:
                                    st.markdown('<div class="warning-card"><h4>🟡 Potential Stage 2 Setup</h4><p>Stock shows promise but needs improvement</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="danger-card"><h4>🔴 Not in Stage 2 Uptrend</h4><p>Stock does not meet trend criteria</p></div>', unsafe_allow_html=True)

                            with col2:
                                st.markdown("**Key Distances:**")
                                for metric, value in distances.items():
                                    color = "green" if value > 0 else "red"
                                    st.markdown(f"**{metric}:** <span style='color: {color}'>{value:+.1f}%</span>", unsafe_allow_html=True)

                    with tab3:
                        st.subheader("📊 Pattern Analysis")

                        col1, col2 = st.columns(2)

                        with col1:
                            # VCP Analysis
                            vcp_analysis = analyzer.detect_vcp_pattern()
                            if vcp_analysis:
                                st.markdown("**Volatility Contraction Pattern (VCP):**")
                                if vcp_analysis['is_vcp']:
                                    st.markdown('<div class="success-card"><h4>✅ VCP Detected</h4><p>Stock shows volatility contraction pattern</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="warning-card"><h4>⚠️ No Clear VCP</h4><p>Pattern needs more development</p></div>', unsafe_allow_html=True)

                                st.write(f"Pivot Highs: {vcp_analysis['pivot_highs']}")
                                st.write(f"Pivot Lows: {vcp_analysis['pivot_lows']}")

                        with col2:
                            # Volume Analysis
                            volume_analysis = analyzer.analyze_volume_pattern()
                            if volume_analysis:
                                st.markdown("**Volume Analysis:**")
                                volume_ratio = volume_analysis['volume_ratio']

                                if volume_ratio > 1.5:
                                    st.markdown('<div class="success-card"><h4>🔊 Above Average Volume</h4><p>Strong institutional interest</p></div>', unsafe_allow_html=True)
                                elif volume_ratio > 0.8:
                                    st.markdown('<div class="warning-card"><h4>📊 Normal Volume</h4><p>Average trading activity</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="danger-card"><h4>🔇 Below Average Volume</h4><p>Lack of institutional support</p></div>', unsafe_allow_html=True)

                                st.write(f"Volume Ratio: {volume_ratio:.2f}x")
                                st.write(f"Recent Avg: {volume_analysis['recent_avg_volume']:,.0f}")
                                st.write(f"50-day Avg: {volume_analysis['50_day_avg_volume']:,.0f}")

                    with tab4:
                        st.subheader("⚖️ Risk/Reward Analysis")

                        # Get current price as default entry
                        current_price = analyzer.data['Close'].iloc[-1]
                        entry_price = st.number_input("Entry Price:", value=float(current_price), step=0.01)

                        risk_reward = analyzer.calculate_risk_reward(entry_price, stop_loss_percent)

                        if risk_reward:
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("**Risk Parameters:**")
                                st.write(f"Entry Price: ${entry_price:.2f}")
                                st.write(f"Stop Loss: ${risk_reward['stop_loss']:.2f}")
                                st.write(f"Risk per Share: ${risk_reward['risk_per_share']:.2f}")
                                st.write(f"Risk %: {risk_reward['risk_percent']:.1f}%")

                            with col2:
                                st.markdown("**Reward Targets:**")
                                st.write(f"2R Target: ${risk_reward['target_2r']:.2f}")
                                st.write(f"3R Target: ${risk_reward['target_3r']:.2f}")
                                st.write(f"Resistance: ${risk_reward['target_resistance']:.2f}")

                            with col3:
                                st.markdown("**Risk/Reward Ratios:**")
                                st.write(f"2R Ratio: {risk_reward['reward_risk_2r']:.1f}:1")
                                st.write(f"3R Ratio: {risk_reward['reward_risk_3r']:.1f}:1")

                                if risk_reward['reward_risk_2r'] >= 2:
                                    st.markdown('<div class="success-card"><p>✅ Good Risk/Reward</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="warning-card"><p>⚠️ Poor Risk/Reward</p></div>', unsafe_allow_html=True)

                    with tab5:
                        st.subheader("💰 Position Sizing Calculator")

                        position_analysis = analyzer.calculate_position_size(
                            account_size, risk_percent, entry_price, stop_loss_percent
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Position Details:**")
                            st.write(f"Recommended Shares: {position_analysis['shares']:,}")
                            st.write(f"Position Value: ${position_analysis['position_value']:,.2f}")
                            st.write(f"Position %: {position_analysis['position_percent']:.1f}%")
                            st.write(f"Risk Amount: ${position_analysis['risk_amount']:.2f}")

                        with col2:
                            st.markdown("**Risk Management:**")

                            # Position size validation
                            if position_analysis['position_percent'] <= 25:
                                st.markdown('<div class="success-card"><p>✅ Position size within limits</p></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="danger-card"><p>❌ Position too large (>25%)</p></div>', unsafe_allow_html=True)

                            # Risk validation
                            if risk_percent <= 2.5:
                                st.markdown('<div class="success-card"><p>✅ Risk within Minervini guidelines</p></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="warning-card"><p>⚠️ High risk per trade</p></div>', unsafe_allow_html=True)

                        # Summary box
                        st.markdown("---")
                        st.markdown("### 📋 Trade Summary")

                        summary_data = {
                            "Metric": ["Entry Price", "Stop Loss", "Shares", "Position Value", "Risk Amount", "2R Target", "3R Target"],
                            "Value": [
                                f"${entry_price:.2f}",
                                f"${risk_reward['stop_loss']:.2f}" if risk_reward else "N/A",
                                f"{position_analysis['shares']:,}",
                                f"${position_analysis['position_value']:,.2f}",
                                f"${position_analysis['risk_amount']:.2f}",
                                f"${risk_reward['target_2r']:.2f}" if risk_reward else "N/A",
                                f"${risk_reward['target_3r']:.2f}" if risk_reward else "N/A"
                            ]
                        }

                        st.table(pd.DataFrame(summary_data))
                else:
                    st.error("Unable to fetch data for the specified ticker. Please check the symbol and try again.")
        else:
            st.warning("Please enter a stock ticker symbol.")

    # Educational section
    with st.expander("📚 About Minervini's SEPA® Methodology"):
        st.markdown("""
        **SEPA® (Specific Entry Point Analysis)** is Mark Minervini's proprietary trading methodology based on:

        **1. Trend Template (Stage 2 Uptrend Criteria):**
        - Stock above 150-day and 200-day moving averages
        - 150-day MA above 200-day MA
        - 200-day MA trending upward
        - Current price within 25% of 52-week high
        - Strong relative strength

        **2. Risk Management:**
        - Maximum 2-2.5% risk per trade
        - Position sizes of 20-25% maximum
        - Stop losses typically 7-8%
        - 2:1 minimum risk/reward ratio

        **3. Entry Patterns:**
        - VCP (Volatility Contraction Pattern)
        - Cup with Handle
        - Primary Base breakouts
        - Power Play setups

        **4. Volume Characteristics:**
        - Above-average volume on breakouts
        - Low volume during base formation
        - Institutional accumulation patterns
        """)

if __name__ == "__main__":
    main()
