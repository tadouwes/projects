import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Portfolio Risk Dashboard", page_icon="📈", layout="wide")

st.title("📈 Portfolio Risk Dashboard")
st.markdown("*A simple risk analytics dashboard*")

# Sidebar
st.sidebar.header("Portfolio Configuration")
portfolio_value = st.sidebar.number_input("Portfolio Value ($M)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0)
confidence_level = st.sidebar.slider("VaR Confidence Level (%)", min_value=90, max_value=99, value=95)
time_horizon = st.sidebar.slider("Time Horizon (days)", min_value=1, max_value=252, value=10)
num_simulations = st.sidebar.slider("Monte Carlo Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
annual_drift = st.sidebar.number_input("Annual Drift (%)", min_value=-10.0, max_value=100.0, value=0.0, step=0.1) / 100

# Generate portfolio data
@st.cache_data
def generate_portfolio_data(n_assets=5):
    assets = ['Equity', 'Fixed Income', 'Commodities', 'FX', 'Alternatives']
    weights = np.random.dirichlet(np.ones(n_assets))
    return pd.DataFrame({
        'Asset Class': assets,
        'Weight (%)': weights * 100,
        'Value ($M)': weights * portfolio_value
    })

# Monte Carlo simulation
@st.cache_data
def run_monte_carlo(portfolio_val, volatility, n_sims, days, drift):
   np.random.seed(42)
    daily_returns = np.random.normal(loc=0, scale=volatility / np.sqrt(252), size=(n_sims, days))
    # CHANGE 'loc=0' to use the drift:
    daily_drift = drift / 252 # Convert annual drift to daily drift
    daily_returns = np.random.normal(loc=daily_drift, scale=volatility / np.sqrt(252), size=(n_sims, days))
    
    # Calculate paths
    cum_returns = np.cumprod(1 + daily_returns, axis=1)
    portfolio_paths = portfolio_val * cum_returns
    
    # Add initial value
    initial_values = np.full((n_sims, 1), portfolio_val)
    portfolio_paths = np.concatenate([initial_values, portfolio_paths], axis=1)
    
    final_values = portfolio_paths[:, -1]
    pnl = final_values - portfolio_val
    return portfolio_paths, pnl

def calculate_risk_metrics(pnl, confidence):
    var = np.percentile(pnl, 100 - confidence)
    cvar = pnl[pnl <= var].mean()
    return var, cvar

# Run simulation
# Run simulation
portfolio_df = generate_portfolio_data()
volatility = 0.15
# CHANGE THIS LINE to include the drift input:
paths, pnl = run_monte_carlo(portfolio_value, volatility, num_simulations, time_horizon)
# TO THIS:
paths, pnl = run_monte_carlo(portfolio_value, volatility, num_simulations, time_horizon, annual_drift) 
var, cvar = calculate_risk_metrics(pnl, confidence_level)

# Metrics
st.subheader("Key Risk Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label=f"VaR ({confidence_level}%, {time_horizon}d)", value=f"${abs(var):.2f}M", delta=f"{(var/portfolio_value)*100:.2f}%", delta_color="inverse")

with col2:
    st.metric(label=f"CVaR ({confidence_level}%, {time_horizon}d)", value=f"${abs(cvar):.2f}M", delta=f"{(cvar/portfolio_value)*100:.2f}%", delta_color="inverse")

with col3:
    st.metric(label="Portfolio Volatility", value=f"{volatility*100:.1f}%", delta="Annual")

with col4:
    expected_return = np.mean(pnl)
    st.metric(label=f"Expected P&L ({time_horizon}d)", value=f"${expected_return:.2f}M")

with col5:
    with col5:
    # Use the variable defined in the sidebar
    st.metric(label="Annualized Drift", value=f"{annual_drift*100:.1f}%", delta="User-Defined Return")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Portfolio", "📉 P&L Distribution", "🎯 VaR Analysis"])

with tab1:
    st.subheader("Portfolio Allocation")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(portfolio_df, values='Weight (%)', names='Asset Class', title='Asset Allocation', hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(portfolio_df.style.format({'Weight (%)': '{:.2f}%', 'Value ($M)': '${:.2f}M'}), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Simulated P&L Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=50, name='P&L Distribution', marker_color='lightblue'))
    fig.add_vline(x=var, line_dash="dash", line_color="red", annotation_text=f"VaR: ${var:.2f}M", annotation_position="top left")
    fig.add_vline(x=cvar, line_dash="dash", line_color="darkred", annotation_text=f"CVaR: ${cvar:.2f}M", annotation_position="bottom left")
    fig.update_layout(title=f"P&L Distribution ({num_simulations:,} simulations)", xaxis_title="P&L ($M)", yaxis_title="Frequency", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean P&L", f"${np.mean(pnl):.2f}M")
    col2.metric("Std Dev", f"${np.std(pnl):.2f}M")
    col3.metric("Worst Case", f"${np.min(pnl):.2f}M")

with tab3:
    st.subheader("Value at Risk Analysis")
    
    # Create figure
    fig = go.Figure()
    
    # Time steps (including t=0)
    time_steps = np.arange(time_horizon + 1)
    
    # Sample paths - reduced to 50 for better visibility
    sample_indices = np.random.choice(num_simulations, min(50, num_simulations), replace=False)
    
    # Add sample paths first (so they're in background)
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=paths[idx, :],
            mode='lines',
            line=dict(width=0.8, color='rgba(200,200,200,0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add percentile lines with new color scheme
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    line_widths = [4, 2, 4, 2, 4]
    labels = ['Worst 5% (VaR)', '25th percentile', 'Median (50%)', '75th percentile', 'Best 5%']
    
    for p, color, width, label in zip(percentiles, colors, line_widths, labels):
        percentile_path = np.percentile(paths, p, axis=0)
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=percentile_path,
            mode='lines',
            name=label,
            line=dict(width=width, color=color)
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Portfolio Value Paths - {num_simulations:,} simulations over {time_horizon} days",
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($M)",
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.info(f"""
    **Chart Explanation:**
    - **Gray lines**: 50 random portfolio paths from {num_simulations:,} simulations
    - **Red line (5th percentile)**: Worst 5% of outcomes - this is your VaR threshold
    - **Yellow line (50th percentile)**: Median outcome - half do better, half do worse
    - **Green line (95th percentile)**: Best 5% of outcomes
    - **Adjust the "Time Horizon" slider** in the sidebar to model different periods (1-252 days)
    """)

st.markdown("---")
st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Simulated data")
