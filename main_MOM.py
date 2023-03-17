import aiohttp
import asyncio
import pandas as pd
from binance.client import Client
from datetime import datetime
import pandas as pd
from binance.client import Client
import json
import time
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sb
# %matplotlib inline
api_key = ''
api_secret = ''
client = Client(api_key, api_secret)
# info = client.futures_exchange_info()
# print(info)
DAYS = '36 day ago UTC'
KLINE_INTERVAL = client.KLINE_INTERVAL_1HOUR
# Resemple period days
RESEMPLE_PERIOD = 1
import requests

def get_binance_futures_tickers():
    url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
    response = requests.get(url)
    data = response.json()
    # print(data)
    futures_tickers = [ticker['symbol'] for ticker in data if 'USDT' in ticker['symbol']]
    return futures_tickers

tickers = get_binance_futures_tickers()
print(tickers)
async def get_data_for_symbol(symbol, session):
    print(f"Getting data for {symbol}")
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={KLINE_INTERVAL}&startTime={DAYS}"
    async with session.get(url) as resp:
        prices = await resp.json()
        for line in prices:
            del line[5:]
#         print(prices)
    df = pd.DataFrame(prices, columns=['date', 'open', 'high', 'low', 'close'])
    df['date'] = pd.to_datetime(df['date'] / 1000, unit='s')
    df = df.set_index('date')
    print(f"Done getting data for {symbol}")
    return df

def modify_df_with_only_close_prices(df, symbol):
    dataframe = pd.DataFrame(df)
    dataframe.drop(['open', 'high', 'low'], axis=1, inplace=True)
    dataframe.rename(columns={'close':symbol}, inplace=True)
    dataframe[symbol] = pd.to_numeric(dataframe[symbol])
    return dataframe

async def process_all_tickers():
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.ensure_future(get_data_for_symbol(symbol, session)) for symbol in tickers]
        results = await asyncio.gather(*tasks)
    return results


def convert_to_yearly_return(period):
    period_to_days = { '1H':365*24, '24H': 365, '48H': 365/2, '96H': 365/4, '72H': 365/3, '120H': 365/5, '144H': 365/6, '168H': 365/7, '192H': 365/8, '216H': 365/9, '240H': 365/10, '20D':365/20, '30D':365/30,}
    return 365 / period_to_days[period]


def resample_prices(close_prices, freq):
    return close_prices.resample(freq, origin='end').last()

def resample_prices2(close_prices, freq):
    return close_prices.resample(freq, origin='end')

def compute_log_returns(prices):
    return np.log(prices) - np.log(prices.shift(1))

def shift_returns(returns, shift_n):
    return returns.shift(shift_n)

def get_top_n(prev_returns, top_n):
    return (prev_returns. rank(axis=1, ascending=False)<=top_n).astype(int)



# asyncio.run(main())



def create_df(df, period, top_n):
    resemple_data = resample_prices(df, period)
    print(resemple_data)
    log_return_table = compute_log_returns(resemple_data)
    print(log_return_table)
    print(log_return_table.isnull().values.any())
    prev_returns = shift_returns(log_return_table, 1)
    lookahead_returns = shift_returns(log_return_table, -1)
    df_long = get_top_n(log_return_table, top_n)
    df_short = get_top_n(-1*log_return_table, top_n)
    best = df_long.iloc[-1].nlargest(top_n)
    print(f"{period} best tokens are:")
    print((log_return_table * 100).iloc[-1].nlargest(top_n))
    best_df = round((log_return_table * 100).iloc[-1].nlargest(top_n), 2)
    worst = df_short.iloc[-1].nlargest(top_n)
    print(f"{period} worst tokens are:")
#     print(df_short.iloc[-1].nlargest(top_n))
    print((log_return_table * 100).iloc[-1].nsmallest(top_n))
    worst_df = round((log_return_table * 100).iloc[-1].nsmallest(top_n), 2)
    return best_df, worst_df
def calculate_vol(df):
    day_data = resample_prices(df, '24H')
    log_return_day = compute_log_returns(day_data)
    ewma_volatility_20 = log_return_day.ewm(span=20).std() * 365**0.5
#     col_name = ewma_volatility_20.columns
#     print(col_name)

    # Rename the column
#     ewma_volatility_20.rename(columns={col_name: '20 days average vol'}, inplace=True)
    print(ewma_volatility_20.iloc[-1])
    ewma_volatility_10 = log_return_day.ewm(span=10).std() * 365**0.5
    print(ewma_volatility_10.iloc[-1])
    ewma_volatility_5 = log_return_day.ewm(span=5).std() * 365**0.5
    print(ewma_volatility_5.iloc[-1])
    return round(ewma_volatility_20.iloc[-1], 2), round(ewma_volatility_10.iloc[-1], 2)
def create_df_sharpe(df, period, top_n):
    int_period = convert_to_yearly_return(period)
    day_data = resample_prices(df, '24H')
    log_return_day = compute_log_returns(day_data)
    ewma_volatility = log_return_day.ewm(span=20).std() * 365**0.5
    vol = ewma_volatility.iloc[-1]
    # print(vol)
    resemple_data = resample_prices(df, period)
    log_return_table = compute_log_returns(resemple_data)
#     prev_returns = shift_returns(log_return_table, 1)
    sharpe = log_return_table * int_period / vol
#     print(sharpe)
    lookahead_returns = shift_returns(log_return_table, -1)
#     df_long = get_top_n(log_return_table, top_n)
#     df_short = get_top_n(-1*log_return_table, top_n)
#     sharpe_best = get_top_n(sharpe, top_n)
#     print((sharpe).iloc[-1].nlargest(top_n))
#     best = df_long.iloc[-1].nlargest(top_n)
#     print(f"{period} best tokens are:")
#     print((log_return_table * 100).iloc[-1].nlargest(top_n))
    best_df = sharpe.iloc[-1].nlargest(top_n)
#     worst = df_short.iloc[-1].nlargest(top_n)
#     print(f"{period} worst tokens are:")
# #     print(df_short.iloc[-1].nlargest(top_n))
#     print((log_return_table * 100).iloc[-1].nsmallest(top_n))
#     print((sharpe).iloc[-1].nsmallest(top_n))
    worst_df = sharpe.iloc[-1].nsmallest(top_n)
    return best_df, worst_df
def find_all(df):
    periods = ['1H', '24H', '48H', '96H', '72H', '120H', '144H', '168H', '192H', '216H', '240H', '20D', '30D']
    list_best_df = []
    list_worst_df = []
    vol, vol10 = calculate_vol(df)
    for i in range(len(periods)):
        best, worst = create_df(df, periods[i], 25)
        list_best_df.append(best)
        list_worst_df.append(worst)
#     print(list_worst_df)
    big_best = pd.concat(list_best_df,axis=1)
    big_best.columns = periods
    big_best.fillna(0, inplace=True)
    big_best['time win'] = (big_best != 0).sum(axis=1)
    big_best = pd.concat([big_best,vol, vol10], axis=1)
    col_name = big_best.columns[-2]
    big_best.rename(columns={col_name: '20 days average vol'}, inplace=True)
    # print(big_best)
    col_name2 = big_best.columns[-1]
    big_best.rename(columns={col_name: '10 days average vol'}, inplace=True)
    big_worst = pd.concat(list_worst_df,axis=1)
    big_worst.columns = periods
    big_worst.fillna(0, inplace=True)
    big_worst['time worst'] = (big_worst != 0).sum(axis=1)
    big_worst = pd.concat([big_worst,vol], axis=1)
    big_worst.rename(columns={col_name: '20 days average vol'}, inplace=True)
    print(big_worst)
    big_best.to_excel('best_performers.xlsx', sheet_name='Position_management')
    big_worst.to_excel('worst_performers.xlsx', sheet_name='Position_management')
def find_all_sharpe(df):
    periods = ['1H', '24H', '48H', '96H', '72H', '120H', '144H', '168H', '192H', '216H', '240H', '20D', '30D']
    list_best_df = []
    list_worst_df = []
    vol, vol10 = calculate_vol(df)
    for i in range(len(periods)):
        best, worst = create_df_sharpe(df, periods[i], 15)
        list_best_df.append(best)
        list_worst_df.append(worst)
#     print(list_worst_df)
    big_best = pd.concat(list_best_df,axis=1)
    big_best.columns = periods
    big_best.fillna(0, inplace=True)
    big_best['time win'] = (big_best != 0).sum(axis=1)
    big_best = pd.concat([big_best,vol], axis=1)
    col_name = big_best.columns[-1]
    big_best.rename(columns={col_name: '20 days average vol'}, inplace=True)
#     print(big_best)
    big_worst = pd.concat(list_worst_df,axis=1)
    big_worst.columns = periods
    big_worst.fillna(0, inplace=True)
    big_worst['time worst'] = (big_worst != 0).sum(axis=1)
    big_worst = pd.concat([big_worst,vol], axis=1)
    big_worst.rename(columns={col_name: '10 days average vol'}, inplace=True)
    print(big_worst)
    big_best.to_excel('best_performers_sharpe.xlsx', sheet_name='Position_management')
    big_worst.to_excel('worst_performers_sharpe.xlsx', sheet_name='Position_management')
# find_all_sharpe()
#
# find_all()

async def main():
    results = await process_all_tickers()
    df_list = [modify_df_with_only_close_prices(df, symbol) for df, symbol in zip(results, tickers)]
    df = pd.concat(df_list, axis=1)
    find_all_sharpe(df)
    find_all(df)
asyncio.run(main())