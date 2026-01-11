import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from app.theme import COLORS, apply_theme

def render_interactive_kline(df: pd.DataFrame, trades: pd.DataFrame = None, title: str = "Price Chart"):
    """
    Renders an interactive candlestick chart with volume and marked trades.
    """
    # Create subplots: 1 row for price, 1 row for volume (optional, can be overlay or separate)
    # Using 2 rows: Price (70%), Volume (30%)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color=COLORS['up_candle'],
        decreasing_line_color=COLORS['down_candle']
    ), row=1, col=1)

    # Moving Averages (Example logic, check if columns exist)
    for period, color_key in [(5, 'ma5'), (10, 'ma10'), (20, 'ma20'), (60, 'ma60'), (120, 'ma120')]:
        col_name = f'ma{period}'
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[col_name], 
                mode='lines', 
                name=f'MA{period}',
                line=dict(color=COLORS[color_key], width=1)
            ), row=1, col=1)
            
    # Volume
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color=[COLORS['volume_up'] if c >= o else COLORS['volume_down'] for c, o in zip(df['close'], df['open'])]
    ), row=2, col=1)
    
    # Trades
    if trades is not None and not trades.empty:
        # Buy Trades
        buys = trades[trades['direction'] == 'LONG']
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index if 'datetime' not in buys.columns else buys['datetime'],
                y=buys['price'],
                mode='markers',
                name='Buy',
                marker=dict(symbol='triangle-up', size=10, color=COLORS['success'], line=dict(width=1, color='white'))
            ), row=1, col=1)
            
        # Sell Trades
        sells = trades[trades['direction'] == 'SHORT']
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index if 'datetime' not in sells.columns else sells['datetime'],
                y=sells['price'],
                mode='markers',
                name='Sell',
                marker=dict(symbol='triangle-down', size=10, color=COLORS['danger'], line=dict(width=1, color='white'))
            ), row=1, col=1)

    # Layout Updates
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title="Volume", row=2, col=1)
    
    apply_theme(fig)
    return fig

def render_equity_curve(df: pd.DataFrame):
    """
    Renders the equity curve with drawdown fill.
    Expects df to have 'balance' or 'equity' column.
    """
    col = 'balance' if 'balance' in df.columns else 'equity'
    if col not in df.columns:
        return go.Figure()

    fig = go.Figure()
    
    # Equity Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col],
        mode='lines',
        name='Capital',
        line=dict(color=COLORS['equity_line'], width=2),
        fill='tozeroy',
        fillcolor='rgba(66, 165, 245, 0.1)' # Very light blue fill
    ))
    
    # Drawdown (Optional: Calculate Drawdown area)
    # Simple logic: High Watermark
    high_water_mark = df[col].cummax()
    drawdown = df[col] - high_water_mark
    
    # If we want to show drawdown as a separate subplot or overlay, logic goes here.
    # For now, just the main equity curve is sufficient for the first version.
    
    fig.update_layout(
        title="Account Equity",
        yaxis_title="Capital",
        xaxis_title="Time",
        showlegend=True
    )
    
    apply_theme(fig)
    return fig
