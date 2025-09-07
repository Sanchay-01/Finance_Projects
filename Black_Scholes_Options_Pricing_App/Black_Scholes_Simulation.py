# Virtual environment is called BSM

import streamlit as st
import pandas as pd
import numpy as np
from numpy import log,sqrt,exp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm

# Page Configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>

/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container{
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: 100%;   /* make full width inside each column */
    margin: 0 auto;
    flex-direction: column;  /* stack label and value vertically */
}

/* Custom Classes for CALL and PUT values */
.metric-call{
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    border-radius: 10px;
}
.metric-put{
    background-color:#ffcccb;
    color: black;
    border-radius: 10px;
}

/* Style for the value text */
.metric-value{
    font-size: 1.5rem;
    font-weight: bold;
    text-align: center;
    margin: 0;
}

/*Style for the label text */
.metric-label{
    font-size: 1rem;
    margin-bottom: 4px;
    text-align: center;
}
</style>
            """,unsafe_allow_html=True)


# The BlackScholes model uses 5 factors to caulcate the price of an option
# 1. Strike Price
# 2. Current Price
# 3. Volatility of the stock (asssumes constant)
# 4. Risk Free Interest rate (assumes constant)
# 5. Time to maturity

#It only works for European Options i.e. those that can be exercised at maturity

class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, volatility: float, interest_rate: float):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatiltiy = volatility
        self.interest_rate = interest_rate
        
    def calculate_price(self):
        time_to_maturiry = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatiltiy
        interest_rate = self.interest_rate
        
        d1 = (
            log(current_price/strike) + (interest_rate  + 0.5 * volatility ** 2) * time_to_maturiry
            )/(volatility * sqrt(time_to_maturiry))
        
        d2 = d1 - (volatility * sqrt(time_to_maturiry))
        
        call_price = current_price * norm.cdf(d1) -(strike * exp(-(interest_rate*time_to_maturiry)) * norm.cdf(d2))        
        
        put_price = (strike * exp(-(interest_rate*time_to_maturiry)) * norm.cdf(-d2)) - current_price * norm.cdf(-d1)
        
        self.call_price = call_price
        self.put_price = put_price
        
        # Greeks
        
        # Delta - rate of change of option w.r.t underlying asset
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)
        
        # Gamma - rate of change delta w.r.t underlying asset
        self.call_gamma = norm.pdf(d1) / (strike * volatility * sqrt(time_to_maturiry))
        self.put_gamma = self.call_gamma
        
        # Theta - rate of change of option price w.r.t time
        self.call_theta = -((current_price*norm.pdf(d1)*volatility)/(2*sqrt(time_to_maturiry)))-interest_rate*strike*exp(-interest_rate*time_to_maturiry)*norm.cdf(d2)
        self.put_theta = -((current_price*norm.pdf(d1)*volatility)/(2*sqrt(time_to_maturiry)))+interest_rate*strike*exp(-interest_rate*time_to_maturiry)*norm.cdf(-d2)
        
        # Vega - rate of change of option price w.r.t volatility
        self.call_vega=current_price*norm.pdf(d1)*sqrt(time_to_maturiry)
        self.put_vega=current_price*norm.pdf(d1)*sqrt(time_to_maturiry)
        
        # Rho - rate of change of option price w.r.t risk-free interest rate
        self.call_rho = strike*time_to_maturiry*exp(-interest_rate*time_to_maturiry)*norm.cdf(d2)
        self.put_rho = -strike*time_to_maturiry*exp(-interest_rate*time_to_maturiry)*norm.cdf(-d2)
        
        return call_price, put_price
    
def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range),len(spot_range)))
    put_prices = np.zeros((len(vol_range),len(spot_range)))
    
    for i, vol in enumerate (vol_range):
        for j, spot in enumerate (spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_price()
            call_prices[i,j] = bs_temp.call_price
            put_prices[i,j] = bs_temp.put_price
    
    fig_call, ax_call = plt.subplots(figsize=(10,8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    fig_put, ax_put = plt.subplots(figsize=(10,8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

    
with st.sidebar:
    
    st.title("📊 Black-Scholes Model")
    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturiry = st.number_input("Time to Maturity(Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=.05)
    
    st.markdown("---")
    calculate_btn=st.button("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Minimum Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Maximum Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min,spot_max,10)
    vol_range = np.linspace(vol_min,vol_max,10)
    
st.title("Black-Scholes Pricing Model")

input_data={
    "Current Asset Price":[current_price],
    "Strike Price":[strike],
    "Time to Maturity (Years)":[time_to_maturiry],
    "Volatility (σ)":[volatility],
    "Risk-Free Interest Rate":[interest_rate],
}

input_df = pd.DataFrame(input_data)
st.table(input_df)

bs_model = BlackScholes(time_to_maturiry, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_price()

col_call,col_put=st.columns([1,1], gap="small")

with col_call:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
        <br>
    """, unsafe_allow_html=True)
    call_greek_cols = st.columns([1,1,1,1,1],gap="small")
    call_values=[bs_model.call_delta,bs_model.call_gamma,bs_model.call_vega,bs_model.call_theta,bs_model.call_rho]
    call_labels=["Delta","Gamma","Vega","Theta","Rho"]
    
    for i,col in enumerate(call_greek_cols):
        with col:
            with col:
                st.markdown(f"""
                    <div class="metric-container metric-call">
                        <div>
                            <div class="metric-label">{call_labels[i]}</div>
                            <div class="metric-value">{call_values[i]:.4f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    
with col_put:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    put_greek_cols = st.columns([1,1,1,1,1],gap="small")
    put_values=[bs_model.put_delta,bs_model.put_gamma,bs_model.put_vega,bs_model.put_theta,bs_model.put_rho]
    put_labels=["Delta","Gamma","Vega","Theta","Rho"]
    
    for i,col in enumerate(put_greek_cols):
        with col:
            with col:
                st.markdown(f"""
                    <div class="metric-container metric-put">
                        <div>
                            <div class="metric-label">{put_labels[i]}</div>
                            <div class="metric-value">{put_values[i]:.4f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)
    
with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)    