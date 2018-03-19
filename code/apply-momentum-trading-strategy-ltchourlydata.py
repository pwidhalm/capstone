# TODO:
# analysis of results (ftest, graphs)
# what did we learn
# model best results

# Code modified and fixed from Stock Technical Analysis with Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
from sklearn.model_selection import ParameterGrid
import time

### PARAMETERS
# default
yr_trading_periods = 365.*24.
trading_frequency = 1
show_graphs = False
print_results = False

# scenarios
# dial in initial numbers and algorithm
# scenario_grid = {'bbperiod':[4],'rsiperiod':[8],'stddev':[1.3],'buy':[35],'sell':[75], 'algorithm':[1,2]}
# show_graphs = True
# print_results = True

# expand before and after numbers with better algorithm
scenario_grid = {'bbperiod': [3, 4, 5],
                 'rsiperiod': [7, 8, 9],
                 'stddev': [1.3, 1.4, 1.5],
                 'buy': [30, 35, 40, 45],
                 'sell': [70, 75, 80],
                 'algorithm': [1]}

# final numbers 1.26 v. 0.83
# scenario_grid = {'bbperiod': [4],
#                  'rsiperiod': [8],
#                  'stddev': [1.3],
#                  'buy': [35],
#                  'sell': [75],
#                  'algorithm': [1]}
show_graphs = False
print_results = False

### DATA
exchange_data = pd.read_csv("./ltchourly-modified.csv")
exchange_data.utc = pd.to_datetime(exchange_data.utc)
exchange_data.index = exchange_data.utc
del exchange_data['utc']
exchange_data = exchange_data.loc['2017-11-20T06:00:00':'2018-02-10T23:59:00']
# exchange_data = exchange_data.loc['2017-11-20T06:00:00':'2017-12-15T23:59:00']
trading_periods = len(exchange_data)


def main():
	'''
	Main
	'''
	print("begin processing scenarios...")
	start = time.clock()
	no_scenarios = 0
	# determine results for each scenario
	results = []
	for idx, scenario in enumerate(list(ParameterGrid(scenario_grid))):
		results.extend(apply_trading_strategy(exchange_data,scenario, idx))
		if show_graphs:
			graph_data(exchange_data, scenario)
		no_scenarios = idx
		if idx % 100==0:
			elapse_time = (time.clock()-start)/60.
			print("%d scenarios processed in %0.2f min" % (no_scenarios, elapse_time))
	print("")
	elapse_time = (time.clock()-start)/60.
	print("...processed %d scenarios in %0.2f min" % (no_scenarios, elapse_time))

	# dataframe
	results_df = pd.DataFrame(results)
	# results
	if print_results:
		grouped_df = results_df.groupby('scenario')
		for key, item in grouped_df:
			with pd.option_context('display.precision',2):
				print(grouped_df.get_group(key), '\n\n')
	# persist df
	results_df.to_csv("./ltchourly-results.csv")
	exchange_data.to_csv("./ltchourly-withsignals.csv")


def apply_trading_strategy(exchange_data,scenario, idx):
	'''
	Apply trading strategy to dataframe for given scenario.

	:param df:
	:param scenario:
	:return: summary dataframe
	'''
	bb_period = scenario['bbperiod']
	bb_stddev = scenario['stddev']
	rsi_period = scenario['rsiperiod']
	rsi_buy = scenario['buy']
	rsi_sell = scenario['sell']

	# bollinger bands
	exchange_data['bb_high'],exchange_data['bb_mid'],exchange_data['bb_low'] = \
		ta.BBANDS(np.asarray(exchange_data.close),timeperiod=bb_period,nbdevup=bb_stddev,nbdevdn=bb_stddev,matype=0)
	# rsi
	exchange_data['rsi'] = ta.RSI(np.asarray(exchange_data.close),timeperiod=rsi_period)

	### TRADING SIGNAL (buy=1 , sell=-1, hold=0)
	# price cross over BB and RSI cross over threshold
	# backteset BB to avoid back-testing bias
	exchange_data['close_lag1'] = exchange_data.close.shift(1)
	exchange_data['vol_lag1'] = exchange_data.vol.shift(1)
	exchange_data['bb_low_lag1'] = exchange_data.bb_low.shift(1)
	exchange_data['bb_mid_lag1'] = exchange_data.bb_mid.shift(1)
	exchange_data['bb_high_lag1'] = exchange_data.bb_high.shift(1)
	exchange_data['close_lag2'] = exchange_data.close.shift(2)
	exchange_data['vol_lag2'] = exchange_data.vol.shift(2)
	exchange_data['bb_low_lag2'] = exchange_data.bb_low.shift(2)
	exchange_data['bb_mid_lag2'] = exchange_data.bb_mid.shift(1)
	exchange_data['bb_high_lag2'] = exchange_data.bb_high.shift(2)
	exchange_data['rsi_lag1'] = exchange_data.rsi.shift(1)
	exchange_data['rsi_lag2'] = exchange_data.rsi.shift(2)

	# generate trading signals
	exchange_data['signal'] = 0  # default to do nothing
	if scenario['algorithm']==1: # bb and rsi
		# if lag2 price is less than lag2 bb lower and the opposite for lag1 values, then buy signal
		exchange_data.loc[(exchange_data.close_lag2<exchange_data.bb_low_lag2) &
		                  (exchange_data.close_lag1<exchange_data.bb_low_lag1) &
		                  (exchange_data.rsi_lag1<rsi_buy),'signal'] = 1
		# if lag2 price is less than lag2 bb upper and the opposite for lag1 values, then sell signal
		exchange_data.loc[(exchange_data.close_lag2>exchange_data.bb_high_lag2) &
		                  (exchange_data.close_lag1>exchange_data.bb_high_lag1) &
		                  (exchange_data.rsi_lag1>rsi_sell),'signal'] = -1
	elif scenario['algorithm']==2: # rsi only
		exchange_data.loc[(exchange_data.rsi_lag2<rsi_buy) &
		                  (exchange_data.rsi_lag1<rsi_buy),'signal'] = 1
		exchange_data.loc[(exchange_data.rsi_lag2>rsi_buy) &
		                  (exchange_data.rsi_lag1>rsi_sell),'signal'] = -1
	else:
		raise "no algorithm provided!"

	# first signal will be a buy
	exchange_data.iloc[0,exchange_data.columns.get_loc('signal')] = 1
	if print_results:
		print(exchange_data.signal.value_counts())

	### TRADING STRATEGY
	# own asset=1, not own asset=0
	exchange_data['portfolio'] = 1
	portfolio = 0
	for i,r in enumerate(exchange_data.iterrows()):
		if r[1]['signal']==1:
			portfolio = 1
		elif r[1]['signal']==-1:
			portfolio = 0
		else:
			portfolio = exchange_data.portfolio[i-1]
		exchange_data.iloc[i,exchange_data.columns.get_loc('portfolio')] = portfolio

	### ANALYSIS
	# Strategies Daily Returns
	# Bands Crossover Strategy Without Trading Commissions
	exchange_data['trade_returns'] = ((exchange_data.close/exchange_data.close_lag1)-1)*exchange_data.portfolio
	exchange_data.iloc[0,exchange_data.columns.get_loc('trade_returns')] = 0.0  # no return for the first period
	# Buy and Hold Strategy
	exchange_data['bh_returns'] = (exchange_data.close/exchange_data.close_lag1)-1
	exchange_data.iloc[0,exchange_data.columns.get_loc('bh_returns')] = 0.0  # no return for the first period

	# cummulative returns
	exchange_data['trade_cum_returns'] = (np.cumprod(exchange_data.trade_returns+1)-1)
	exchange_data['bh_cum_returns'] = (np.cumprod(exchange_data.bh_returns+1)-1)

	# Strategies Performance Metrics
	# Annualized Returns
	trade_yr_returns = (1+exchange_data.trade_cum_returns.tail(1).values[0])**(yr_trading_periods/trading_periods)-1
	bh_yr_returns = (1+exchange_data.bh_cum_returns.tail(1).values[0])**(yr_trading_periods/trading_periods)-1
	# Annualized Standard Deviation
	trade_yr_std = exchange_data.trade_returns.std()*np.sqrt(yr_trading_periods)  # cryptos trade
	bh_yr_std = exchange_data.bh_returns.std()*np.sqrt(yr_trading_periods)
	# Annualized Sharpe Ratio
	trade_yr_sharpe = np.sqrt(yr_trading_periods)*exchange_data.trade_returns.mean()/exchange_data.trade_returns.std()
	bh_yr_sharpe = np.sqrt(yr_trading_periods)*exchange_data.bh_returns.mean()/exchange_data.bh_returns.std()

	# Summary Results Data Table
	results = [{'scenario':idx,'strategy':'trade','return':trade_yr_returns,'stddev':trade_yr_std,'sharpe':trade_yr_sharpe,
	            'bbperiod': scenario['bbperiod'],'rsiperiod':scenario['rsiperiod'],'stddev': scenario['stddev'],
	            'buy':scenario['buy'],'sell':scenario['sell'],'algorithm':scenario['algorithm']},
	           {'scenario':idx,'strategy':'buy&hold','return':bh_yr_returns,'stddev':bh_yr_std,'sharpe':bh_yr_sharpe,
	            'bbperiod':scenario['bbperiod'],'rsiperiod':scenario['rsiperiod'],'stddev':scenario['stddev'],
	            'buy':scenario['buy'],'sell':scenario['sell'],'algorithm':scenario['algorithm']}]
	return results


def graph_data(exchange_data, scenario):
	'''
	graph data
	:param df:
	:return:
	'''
	bb_timeperiod = scenario['bbperiod']
	bb_stddev = scenario['stddev']
	rsi_timeperiod = scenario['rsiperiod']
	rsi_buy = scenario['buy']
	rsi_sell = scenario['sell']

	fig1,ax = plt.subplots(5,sharex=True)
	ax[0].plot(exchange_data['close'])
	ax[0].plot(exchange_data['bb_high'],linestyle='--',label='high')
	ax[0].plot(exchange_data['bb_mid'],linestyle='--',label='middle')
	ax[0].plot(exchange_data['bb_low'],linestyle='--',label='low')
	ax[0].legend(loc='upper left')
	ax[1].plot(exchange_data['rsi'],color='green',label='rsi')
	ax[1].axhline(y=rsi_sell,linestyle='--',color='orange')
	ax[1].axhline(y=rsi_buy,linestyle='--',color='orange')
	ax[1].legend(loc='upper left')
	ax[2].plot(exchange_data['signal'],marker='o',markersize=5,linestyle='',label='signal',color='red')
	ax[2].legend(loc='upper left')
	ax[3].plot(exchange_data['portfolio'],marker='o',markersize=5,linestyle='',label='portfolio',color='green')
	ax[3].legend(loc='upper left')
	ax[4].plot(exchange_data['trade_cum_returns'],label='trade')
	ax[4].plot(exchange_data['bh_cum_returns'],label='buy&hold')
	ax[4].legend(loc='upper left')
	plt.suptitle(
			'BTC %shr Close Prices, BB (%s, %s), & RSI (%s)'%(trading_frequency,bb_timeperiod,bb_stddev,rsi_timeperiod))
	plt.show()


if __name__=='__main__':
	main()
