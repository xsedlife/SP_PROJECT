import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Portfel Inwestycyjny", layout="wide")

st.title("📊 Ranking inwestycyjny")

@st.cache_data(ttl=3600)

def load_data():
    df_data = pd.read_excel("data.xlsx",sheet_name="stocks")
    df_tickers = pd.read_excel("data.xlsx",sheet_name="tickers")
    return df_data, df_tickers

def get_price_change(ticker, date=None):
    stock = yf.Ticker(ticker)

    hist = stock.history(period="1y")
    if hist.empty:
        return 'Niepoprawny ticker'
    
    last_price = hist['Close'].iloc[-1] 
    
    if date == 'rok do roku':
        price = hist['Close'].iloc[0] 
    else:
        price = hist.loc[hist.index.year == 2025]['Close'].iloc[-1]
        
    return_rate = (last_price - price) / price 

    return return_rate

def edit_df_data(df, tickers, date=None):
    df = df.copy()
    robot_users = ['Claude Sonnet', 'Google Gemini', 'Chat GPT']
    df['name'] = np.where(
        df['name'].isin(robot_users),
        df['name'] + ' 👾',
        df['name']
    )

    rr_cols = []
    rr_spolki = []
    unique_tickers = tickers['ticker'].unique()
    price_cache = {}
    for ticker in unique_tickers:
        if date is None:
            price_cache[ticker] = get_price_change(ticker)
        elif date == 'rok do roku':
            price_cache[ticker] = get_price_change(ticker, date=date)

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
        if c in ['pl','usa','world']:
            rr_spolki.append(new_name)
        df[new_name] = df[c].map(price_cache) * 100

    mapping_dict = tickers.set_index('ticker')['t_name'].to_dict()
    df[columns] = df[columns].replace(mapping_dict)

    df['return rate'] = df[rr_cols].mean(axis=1)
    
    df['spolki_rr'] = df[rr_spolki].mean(axis=1)

    df = df.sort_values(by='return rate',ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    df['name'] = np.where(
        df.index == 0, '🥇 ' + df['name'], np.where(
            df.index == 1, '🥈 ' + df['name'], np.where(
               df.index == 2, '🥉' + df['name'], df['name'])))


    return df

def heat_map(val, benchmark):
    if val > 0 and val > benchmark:
        return 'background-color: #006B07; color: white' 
    elif 0 <= val <= benchmark:
        return 'background-color: #737100; color: white' 
    elif val < 0:
        return 'background-color: #801E00; color: white' 
    return ''


############################################### <- kod
df_data, df_tickers = load_data()
df_main = edit_df_data(df_data,df_tickers)
df_rok_do_roku = edit_df_data(df_data,df_tickers, date='rok do roku')

sp500 = get_price_change('SPY') * 100
sp500_rok_do_roku = get_price_change('SPY',date = 'rok do roku') * 100

cols_to_style = [col for col in df_main.columns if col.endswith('_rr')] + ['return rate']


styled_df_main = df_main.style.map(
    heat_map,
    benchmark = sp500,
    subset = cols_to_style
)
styled_df_rok_do_roku = df_rok_do_roku.style.map(
    heat_map,
    benchmark = sp500,
    subset = cols_to_style
)


############################################### <- Streamlit
widok = st.segmented_control(
    'Porównanie:',
    options=['Początek roku','Rok do roku'],
    default='Początek roku'
)

if widok == 'Początek roku':
    st.subheader(f'Benchmark SP500: {sp500:.1f}%')
    st.dataframe(
        styled_df_main,
        column_order=(
            'rank',
            'name',
            'pl',
            'pl_rr',
            'usa',
            'usa_rr',
            'world',
            'world_rr',
            'crypto',
            'crypto_rr',
            'commodity',
            'commodity_rr',
            'spolki_rr',
            'return rate'
            ),
        column_config={
            'rank': st.column_config.NumberColumn('Rank'),
            'name': st.column_config.TextColumn('User'),
            'pl_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'usa_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'world_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'crypto_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'commodity_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'pl': st.column_config.TextColumn('Polska'),
            'usa': st.column_config.TextColumn('USA'),
            'world': st.column_config.TextColumn('Świat'),
            'crypto': st.column_config.TextColumn('Krypto'),
            'commodity': st.column_config.TextColumn('Surowiec'),
            'spolki_rr': st.column_config.NumberColumn('Zwrot (spółki)', format='%.1f %%'),
            'return rate': st.column_config.NumberColumn('Stopa zwrotu', format='%.1f %%')
        },
        width='stretch',
        hide_index=True,
        )

    wygraniec, frajer = st.columns(2)
    with wygraniec:
        st.markdown(f"<h3 style='text-align: center;'>KOKS TYGODNIA:<br>{df_main['name'].iloc[0]}</h3>", unsafe_allow_html=True)

        st.image("gifs/jasperkasiorka-dawid-jasper.gif", width='stretch')

    with frajer:
        st.markdown(f"<h3 style='text-align: center;'>FRAJER TYGODNIA:<br>{'🫵🏼🤣 ' + df_main['name'].iloc[-1]}</h3>", unsafe_allow_html=True)

        st.image("gifs/dawid.gif", width='stretch')
else:
    st.subheader(f'Benchmark SP500: {sp500_rok_do_roku:.1f}%')
    st.dataframe(
        styled_df_rok_do_roku,
        column_order=(
            'rank',
            'name',
            'pl',
            'pl_rr',
            'usa',
            'usa_rr',
            'world',
            'world_rr',
            'crypto',
            'crypto_rr',
            'commodity',
            'commodity_rr',
            'spolki_rr',
            'return rate'
            ),
        column_config={
            'rank': st.column_config.NumberColumn('Rank'),
            'name': st.column_config.TextColumn('User'),
            'pl_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'usa_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'world_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'crypto_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'commodity_rr': st.column_config.NumberColumn('%', format='%.1f %%'),
            'pl': st.column_config.TextColumn('Polska'),
            'usa': st.column_config.TextColumn('USA'),
            'world': st.column_config.TextColumn('Świat'),
            'crypto': st.column_config.TextColumn('Krypto'),
            'commodity': st.column_config.TextColumn('Surowiec'),
            'spolki_rr': st.column_config.NumberColumn('Zwrot (spółki)', format='%.1f %%'),
            'return rate': st.column_config.NumberColumn('Stopa zwrotu', format='%.1f %%')
        },
        width='stretch',
        hide_index=True,
        )

    wygraniec, frajer = st.columns(2)
    with wygraniec:
        st.markdown(f"<h3 style='text-align: center;'>KOKS ROKU:<br>{df_rok_do_roku['name'].iloc[0]}</h3>", unsafe_allow_html=True)

        st.image("gifs/jasperkasiorka-dawid-jasper.gif", width='stretch')

    with frajer:
        st.markdown(f"<h3 style='text-align: center;'>FRAJER ROKU:<br>{'🫵🏼🤣 ' + df_rok_do_roku['name'].iloc[-1]}</h3>", unsafe_allow_html=True)

        st.image("gifs/dawid.gif", width='stretch')