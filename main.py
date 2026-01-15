import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Portfel Inwestycyjny", layout="wide")

st.title("📊 Mój Portfel Inwestycyjny")

@st.cache_data(ttl=3600)

def load_data():
    df_data = pd.read_excel("data.xlsx",sheet_name="stocks")
    df_tickers = pd.read_excel("data.xlsx",sheet_name="tickers")
    return df_data, df_tickers

def get_price_change(ticker):
    stock = yf.Ticker(ticker)

    hist = stock.history(period="1y")
    if hist.empty:
        return 'Niepoprawny ticker'
    
    # Ostatnia cena (najnowsza dostępna)
    last_price = hist['Close'].iloc[-1] 

    year_price = hist.loc[hist.index.year == 2025]['Close'].iloc[-1]

    return_rate = (last_price - year_price) / year_price 
    return return_rate

def edit_df_data(df, tickers):
    rr_cols = []
    unique_tickers = tickers['ticker'].unique()
    price_cache = {}
    for ticker in unique_tickers:
        price_cache[ticker] = get_price_change(ticker)

    columns = [
        'pl',
        'usa',
        'world',
        'crypto',
        'commodity'
    ]
    for c in columns:
        new_name = c + '_rr'
        rr_cols.append(new_name)
        df[new_name] = df[c].map(price_cache)

    mapping_dict = tickers.set_index('ticker')['t_name'].to_dict()
    df[columns] = df[columns].replace(mapping_dict)

    df['return rate'] = df[rr_cols].mean(axis=1)


    return df


df_data, df_tickers = load_data()
df_main = edit_df_data(df_data,df_tickers)

st.dataframe(df_main)
