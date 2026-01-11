import plotly.graph_objects as go
import plotly.io as pio

# --- Color Palette ---
COLORS = {
    "primary": "#2E86C1",
    "secondary": "#A93226",
    
    # Semantic Colors
    "success": "#27AE60",
    "warning": "#F39C12",
    "danger": "#C0392B",
    "info": "#2980B9",
    
    # Backgrounds
    "bg_dark": "#0E1117",
    "bg_card": "#1E2329",
    
    # Text
    "text_primary": "#FAFAFA",
    "text_secondary": "#B0B3B8",
    
    # Chart Specific
    "up_candle": "#26A69A",   # Green for up
    "down_candle": "#EF5350", # Red for down
    "volume_up": "rgba(38, 166, 154, 0.5)",
    "volume_down": "rgba(239, 83, 80, 0.5)",
    
    # Moving Averages / Lines
    "ma5": "#E1BEE7",  # Light Purple
    "ma10": "#9575CD", # Purple
    "ma20": "#4DB6AC", # Teal
    "ma60": "#FFD54F", # Amber
    "ma120": "#FF8A65", # Deep Orange
    
    # Equity Curve
    "equity_line": "#42A5F5",
    "drawdown_fill": "rgba(239, 83, 80, 0.2)",
    
    # Grid
    "grid": "#2E333D"
}

# --- Plotly Template ---
def get_plotly_template():
    """
    Returns a custom Dark Pro Plotly template.
    """
    layout = go.Layout(
        plot_bgcolor=COLORS["bg_dark"],
        paper_bgcolor=COLORS["bg_dark"],
        font=dict(
            family="Inter, Roboto, Arial, sans-serif",
            color=COLORS["text_secondary"],
            size=12
        ),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            showgrid=True,
            showline=True,
            linecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            showgrid=True,
            showline=True,
            linecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        # Legend styling
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["text_secondary"]), # Slightly smaller font
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Margin adjustments
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified"
    )
    
    return go.layout.Template(layout=layout)

# Defines a simple function to apply the theme to a figure
def apply_theme(fig):
    fig.update_layout(template=get_plotly_template())
    return fig
