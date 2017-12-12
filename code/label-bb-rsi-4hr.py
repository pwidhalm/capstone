# Code modified and fixed from Stock Technical Analysis with Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta

### DATA
df = pd.read_csv("./coinbaseBTCUSD_1min_2014-12-01_to_2017-10-20.csv")
df.timestamp = pd.to_datetime(df.timestamp,unit='s')  # timestamp is in seconds
df.index = df.timestamp
del df['timestamp']
df = df.loc['2014-12-01T06:00:00':'2017-10-19T23:59:00']  # remove rows that do no lie within the hour window

# resample to hourly data to daily
df = df.resample(rule="240T").agg(
		{'open':'first','high':'max','low':'min','close':'last','volbtc':'sum','volusd':'sum','wtdprice':'last'})
# 2017- data
df = df['6-1-2016':].copy()
# bollinger bands
df['bb_high'],df['bb_mid'],df['bb_low'] = ta.BBANDS(np.asarray(df.close),
                                                  timeperiod=7,nbdevup=1.3,nbdevdn=1.3,matype=0)
# rsi
df['rsi'] = ta.RSI(np.asarray(df.close),timeperiod=14)

### TRADING SIGNAL (buy=1 , sell=-1, hold=0)
# price cross over BB and RSI cross over threshold
# backteset BB to avoid back-testing bias
df['close_lag1'] = df.close.shift(1)
df['bb_low_lag1'] = df.bb_low.shift(1)
df['bb_mid_lag1'] = df.bb_mid.shift(1)
df['bb_high_lag1'] = df.bb_high.shift(1)
df['close_lag2'] = df.close.shift(2)
df['bb_low_lag2'] = df.bb_low.shift(2)
df['bb_mid_lag2'] = df.bb_mid.shift(1)
df['bb_high_lag2'] = df.bb_high.shift(2)
df['rsi_lag1'] = df.rsi.shift(1)
df['rsi_lag2'] = df.rsi.shift(2)

# generate trading signals
df['signal'] = 0  # default to do nothing
# TODO: refine until the signals look right!!!!
# if lag2 price is less than lag2 bb lower and the oppostive for lag1 values, then buy signal
df.loc[(df.close_lag2<df.bb_low_lag2) & (df.close_lag1<df.bb_low_lag1) & (df.rsi_lag1<45),'signal'] = 1
# if lag2 price is less than lag2 bb upper and the oppostite for lag1 values, then sell signal
# TODO: need to add a check for enough profit before selling???
df.loc[(df.close_lag2>df.bb_high_lag2) & (df.close_lag1>df.bb_high_lag1) & (df.rsi_lag1>85),'signal'] = -1
# first signal will be a buy
df.iloc[0,df.columns.get_loc('signal')] = 1

print(df.signal.value_counts())

### TRADING STRATEGY
# own asset=1, not own asset=0
df['portfolio'] = 1
portfolio = 0
for i,r in enumerate(df.iterrows()):
	if r[1]['signal']==1:
		portfolio = 1
	elif r[1]['signal']==-1:
		portfolio = 0
	else:
		portfolio = df.portfolio[i-1]
	df.iloc[i,df.columns.get_loc('portfolio')] = portfolio

### ANALYSIS
# Strategies Daily Returns
# Bands Crossover Strategy Without Trading Commissions
df['trade_returns'] = ((df.close/df.close_lag1)-1)*df.portfolio
df.iloc[0,df.columns.get_loc('trade_returns')] = 0.0  # no return for the first period
# Buy and Hold Strategy
df['bh_returns'] = (df.close/df.close_lag1)-1
df.iloc[0,df.columns.get_loc('bh_returns')] = 0.0  # no return for the first period

# Strategies Cumulative Returns
# Cumulative Returns Calculation
# TODO: check calculations
df['trade_cum_returns'] = (np.cumprod(df.trade_returns+1)-1)
df['bh_cum_returns'] = (np.cumprod(df.bh_returns+1)-1)

# Strategies Performance Metrics
# Annualized Returns
trade_yr_returns = df.trade_cum_returns.tail(1).values[0]
bh_yr_returns = df.bh_cum_returns.tail(1).values[0]
# Annualized Standard Deviation
trade_std = np.std(df.trade_returns.values)*np.sqrt(365.)  # cryptos trade 365
bh_std = np.std(df.bh_returns.values)*np.sqrt(365.)
# Annualized Sharpe Ratio
trade_sharpe = trade_yr_returns/trade_std
bh_sharpe = bh_yr_returns/bh_std

# Summary Results Data Table
print('\n')
summary_df = pd.DataFrame(
		{'Summary' :['Return','Std Dev','Sharpe (Rf=0%)'],'Trade':[trade_yr_returns,trade_std,trade_sharpe],
		 'Buy&Hold':[bh_yr_returns,bh_std,bh_sharpe]})
summary_df = summary_df[['Summary','Trade','Buy&Hold']]
with pd.option_context('display.precision',2):
	print(summary_df)

# CHARTING
fig1,ax = plt.subplots(5,sharex=True)
ax[0].plot(df['close'])
ax[0].plot(df['bb_high'],linestyle='--',label='high')
ax[0].plot(df['bb_mid'],linestyle='--',label='middle')
ax[0].plot(df['bb_low'],linestyle='--',label='low')
ax[0].legend(loc='upper left')
ax[1].plot(df['rsi'],color='green',label='rsi')
ax[1].axhline(y=85,linestyle='--',color='orange')
ax[1].axhline(y=45,linestyle='--',color='orange')
ax[1].legend(loc='upper left')
ax[2].plot(df['signal'],marker='o',markersize=5,linestyle='',label='signal',color='red')
ax[2].legend(loc='upper left')
ax[3].plot(df['portfolio'],marker='o',markersize=5,linestyle='',label='portfolio',color='green')
ax[3].legend(loc='upper left')
ax[4].plot(df['trade_cum_returns'],label='trade')
ax[4].plot(df['bh_cum_returns'],label='buy&hold')
ax[4].legend(loc='upper left')
plt.suptitle('BTC 4hr Close Prices, BB (7, 1.3), & RSI (14)')
plt.show()

# # additional indicators
# df['cci'] = ta.CCI(np.asarray(df['high']), np.asarray(df['low']), np.asarray(df['close']), timeperiod=7)
# df['roc'] = ta.ROC(np.asarray(df['close']), timeperiod=21)
# df['rsi2'] = ta.RSI(np.asarray(df.close),timeperiod=2)
# df['rsi14'] = ta.RSI(np.asarray(df.close),timeperiod=14)
# df['cci2'] = ta.CCI(np.asarray(df['high']), np.asarray(df['low']), np.asarray(df['close']), timeperiod=3)
#
#
# fig1,ax = plt.subplots(2,sharex=True)
# ax[0].plot(df['close'])
# ax[0].legend(loc='upper left')
# ax[1].plot(df['cci'],color='green')
# ax[1].plot(df['roc'],color='red')
# ax[1].legend(loc='upper left')
# plt.suptitle('BTC 2hr Close Prices, CCI(7), ROC(7)')
# plt.show()


# persist df
df.to_csv("./coinbaseBTCUSD-withsignals-4hr.csv")
