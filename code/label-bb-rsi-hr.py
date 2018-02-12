# TODO:
# expand date range
# adjust parameters
# keeps track of results
# what did we learn
# model best results

# Code modified and fixed from Stock Technical Analysis with Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta

### PARAMETERS
bb_timeperiod=8
bb_stddev=1.3
rsi_timeperiod=8
rsi_buy=35
rsi_sell=80
yr_trading_periods=365.*24.
trading_frequency=1

### DATA
df = pd.read_csv("./btchourly-modified.csv")
df.utc = pd.to_datetime(df.utc)
df.index = df.utc
del df['utc']
# df = df.loc['2017-11-20T06:00:00':'2018-02-10T23:59:00']
df = df.loc['2017-11-20T06:00:00':'2017-12-15T23:59:00']
trading_periods = len(df)


# bollinger bands
df['bb_high'],df['bb_mid'],df['bb_low'] = ta.BBANDS(np.asarray(df.close),
													timeperiod=bb_timeperiod,nbdevup=bb_stddev, nbdevdn=bb_stddev,
													matype=0)
# rsi
df['rsi'] = ta.RSI(np.asarray(df.close),timeperiod=rsi_timeperiod)

### TRADING SIGNAL (buy=1 , sell=-1, hold=0)
# price cross over BB and RSI cross over threshold
# backteset BB to avoid back-testing bias
df['close_lag1'] = df.close.shift(1)
df['vol_lag1'] = df.vol.shift(1)
df['bb_low_lag1'] = df.bb_low.shift(1)
df['bb_mid_lag1'] = df.bb_mid.shift(1)
df['bb_high_lag1'] = df.bb_high.shift(1)
df['close_lag2'] = df.close.shift(2)
df['vol_lag2'] = df.vol.shift(2)
df['bb_low_lag2'] = df.bb_low.shift(2)
df['bb_mid_lag2'] = df.bb_mid.shift(1)
df['bb_high_lag2'] = df.bb_high.shift(2)
df['rsi_lag1'] = df.rsi.shift(1)
df['rsi_lag2'] = df.rsi.shift(2)

# generate trading signals
df['signal'] = 0  # default to do nothing
# TODO: refine until the signals look right!!!!
# if lag2 price is less than lag2 bb lower and the opposite for lag1 values, then buy signal
df.loc[(df.close_lag2<df.bb_low_lag2) & (df.close_lag1<df.bb_low_lag1) & (df.rsi_lag1<rsi_buy),'signal'] = 1
# if lag2 price is less than lag2 bb upper and the opposite for lag1 values, then sell signal
# TODO: need to add a check for enough profit before selling???
df.loc[(df.close_lag2>df.bb_high_lag2) & (df.close_lag1>df.bb_high_lag1) & (df.rsi_lag1>rsi_sell),'signal'] = -1
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

# cummulative returns
df['trade_cum_returns'] = (np.cumprod(df.trade_returns+1)-1)
df['bh_cum_returns'] = (np.cumprod(df.bh_returns+1)-1)

# Strategies Performance Metrics
# Annualized Returns
trade_yr_returns = (1+df.trade_cum_returns.tail(1).values[0])**(yr_trading_periods/trading_periods) - 1
bh_yr_returns = (1+df.bh_cum_returns.tail(1).values[0])**(yr_trading_periods/trading_periods) - 1
# Annualized Standard Deviation
trade_std = df.trade_returns.std()*np.sqrt(yr_trading_periods)  # cryptos trade
bh_std = df.bh_returns.std()*np.sqrt(yr_trading_periods)
# Annualized Sharpe Ratio
trade_sharpe = np.sqrt(yr_trading_periods)*df.trade_returns.mean() / df.trade_returns.std()
bh_sharpe = np.sqrt(yr_trading_periods)*df.bh_returns.mean() / df.bh_returns.std()


# Summary Results Data Table
print('\n')
summary_df = pd.DataFrame(
		{'Summary' :['Return','Std Dev','Sharpe (Rf=0%)'],
		 'Trade':[trade_yr_returns,trade_std,trade_sharpe],
		 'Buy&Hold':[bh_yr_returns,bh_std,bh_sharpe]})
summary_df = summary_df[['Summary','Trade','Buy&Hold']]
with pd.option_context('display.precision',2):
	print('Trading periods for this dataset: %s' % trading_periods)
	print(summary_df)

# CHARTING
fig1,ax = plt.subplots(5,sharex=True)
ax[0].plot(df['close'])
ax[0].plot(df['bb_high'],linestyle='--',label='high')
ax[0].plot(df['bb_mid'],linestyle='--',label='middle')
ax[0].plot(df['bb_low'],linestyle='--',label='low')
ax[0].legend(loc='upper left')
ax[1].plot(df['rsi'],color='green',label='rsi')
ax[1].axhline(y=rsi_sell,linestyle='--',color='orange')
ax[1].axhline(y=rsi_buy,linestyle='--',color='orange')
ax[1].legend(loc='upper left')
ax[2].plot(df['signal'],marker='o',markersize=5,linestyle='',label='signal',color='red')
ax[2].legend(loc='upper left')
ax[3].plot(df['portfolio'],marker='o',markersize=5,linestyle='',label='portfolio',color='green')
ax[3].legend(loc='upper left')
ax[4].plot(df['trade_cum_returns'],label='trade')
ax[4].plot(df['bh_cum_returns'],label='buy&hold')
ax[4].legend(loc='upper left')
plt.suptitle('BTC %shr Close Prices, BB (%s, %s), & RSI (%s)' % (trading_frequency, bb_timeperiod, bb_stddev,
																 rsi_timeperiod))
plt.show()

# persist df
df.to_csv("./btchourly-withsignals.csv")
