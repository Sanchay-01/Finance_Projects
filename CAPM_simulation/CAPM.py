#environment name is CAPM_env
#source CAPM_env/bin/activate

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
print("Imports Successful!")

def normalise_by_initial_price(df):
    x=df.copy()
    for i in x.columns[1:]:
        x[i]=x[i]/x[i][0]
    return x

def interactive_plot(df,title):
    fig=px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'],y=df[i],name=i)
    fig.show()
    
def daily_return(df):
    df_daily_return=df.copy()
    for i in df_daily_return.columns[1:]:
        for j in range (1,len(df)):
            df_daily_return[i][j]=((df[i][j]-df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return
print("Functions Ready!")

tickers=['HOOD','DELL','QBTS','AUR', 'KOPN','UEC','LCID','INTC','LUNR','ARM','NVDA','META','^SPX']
stock_data=yf.download(tickers,start='2023-09-14',end='2025-08-31')["Close"]
stock_data.to_csv("Stock_data.csv")
stock_data=stock_data.reset_index()
print("CSV created")

interactive_plot(normalise_by_initial_price(stock_data),'Normalised Prices')

stock_daily_return_data=daily_return(stock_data)
stock_daily_return_data.to_csv("Stock_Daily_Returns.csv")
print("CSV created")

# Calculate beta and alpha of the stock
# Beta refers to the extra volatility relative to a benchmark
# Alpha refers to the excess return on an investment after adjusting
# market-related volatility and random fluctuations 
beta, alpha=np.polyfit(stock_daily_return_data['^SPX'],stock_daily_return_data['HOOD'],1)
print('Beta for {} stock is = {} and alpha is = {}'.format('HOOD',beta,alpha))

# Calculate market return
rm = stock_daily_return_data['^SPX'].mean()*252
print("!!!")
print(rm)

# Assume a risk-free rate of 0
rf = 4

# Expected return
re = rf + (beta*(rm-rf))

print(re)

beta = {}
alpha = {}

for i in stock_daily_return_data.columns:
    if i != 'Date' and i !='^SPX':
        stock_daily_return_data.plot(kind = 'scatter', x = '^SPX', y=i)
        b, a = np.polyfit(stock_daily_return_data['^SPX'],stock_daily_return_data[i],1)
        plt.plot(stock_daily_return_data['^SPX'],b*stock_daily_return_data['^SPX']+a,'-',color='r')
        beta[i]=b
        alpha[i]=a
        #plt.show()
        
keys = list (beta.keys())

expected_return={}

for i in keys:
    expected_return[i]=round(rf + (beta[i]*(rm-rf)),2)
    
for i in keys:
    print('Expected Return Based on CAPM for {} is {}%'.format(i,expected_return[i]))

portfolio_weights=1/12 * np.ones(12)
expected_portfolio_return=sum(list(expected_return.values())*portfolio_weights)
print('Expected Return Based on CAPM for the portfolio is {}%\n'.format(expected_portfolio_return))        