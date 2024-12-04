# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:23:28 2024

@author: Amit Bhanja
"""
import yfinance as yf
import pandas as pd
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-v0_8-darkgrid')

significance_levels = {99 : '1%', 95 : '5%', 90 : '10%'}

class Pairs:
    
    def __init__(self, stock1, stock2, start_date, end_date, leverage=1):
        self.stock1 : str = stock1
        self.stock2 : str = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.halfLife = -1
        self.leverage = leverage
        
    def collect_data(self):
        ticker1 = yf.Ticker(self.stock1)
        df1 = ticker1.history(start=self.start_date, end=self.end_date)
        ticker2 = yf.Ticker(self.stock2)
        df2 = ticker2.history(start=self.start_date, end=self.end_date)
        if df1.empty or df2.empty:
            raise Exception("Could not read data for either of the stocks")
        close1 = df1['Close']
        close2 = df2['Close']
        self.df = pd.concat([close1, close2], axis=1)
        self.df.columns = [self.stock1, self.stock2]
        self.df.index = pd.to_datetime(self.df.index)
        
    def findOptimumSpread(self, significance_level):
        if significance_level not in significance_levels:
            raise ValueError(f'Incorrect significance level {significance_level} given')
        n_rows = int(len(self.df) * 0.9)
        self.df = self.df[[self.stock1, self.stock2]].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(self.df) < n_rows or self.df.empty:
            print(f"Insufficient number of rows for {self.stock1} and {self.stock2}")
            return False
        self.model = OLS(self.df[self.stock1].iloc[:n_rows], self.df[self.stock2].iloc[:n_rows])
        self.model = self.model.fit()
        print(f"The hedge ratio for {self.stock1} : {self.stock2} = {self.model.params[0]}")
        self.df['spread'] = self.df[self.stock1] - self.model.params[0] * self.df[self.stock2]
        
        adf = adfuller(self.df.spread)
        print(f"The T-statistic {round(adf[0], 2)} Critical Values {adf[4]}")
        return adf[0] < adf[4][significance_levels[significance_level]]
    
    def half_life(self):
        if self.halfLife != -1:
            return self.halfLife
        spread_x = np.mean(self.df.spread) - self.df.spread
        spread_y = self.df.spread.shift(-1) - self.df.spread
        spread_df = pd.DataFrame({'x':spread_x,'y':spread_y})
        spread_df = spread_df.dropna()
        
        model_s = OLS(spread_df['y'], spread_df['x'])
        model_s = model_s.fit()
        theta = model_s.params[0]
        halfLife = abs(np.log(2) / theta)
        self.halfLife = halfLife
        return halfLife
    
    def mean_reversion_strategy(self):
        lookback_period = int(self.half_life())
        print(f"Mean Reversion strategy for {self.stock1} and {self.stock2} Half Life {lookback_period}")
        self.df['moving_average'] = self.df.spread.rolling(lookback_period).mean()
        self.df['moving_std_dev'] = self.df.spread.rolling(lookback_period).std()
        
        self.df['upper_band'] = self.df.moving_average + 2 * self.df.moving_std_dev
        self.df['lower_band'] = self.df.moving_average - 2 * self.df.moving_std_dev
        
        self.df['long_entry'] = self.df.spread < self.df.lower_band
        self.df['long_exit'] = self.df.spread >= self.df.moving_average
        
        # Long Positions
        self.df['positions_long'] = np.nan
        self.df.loc[self.df.long_entry, 'positions_long'] = 1
        self.df.loc[self.df.long_exit, 'positions_long'] = 0
        self.df.positions_long = self.df.positions_long.fillna(method='ffill')
        
        # Short Positions
        self.df['short_entry'] = self.df.spread > self.df.upper_band
        self.df['short_exit'] = self.df.spread <= self.df.moving_average
        
        self.df['positions_short'] = np.nan
        self.df.loc[self.df.short_entry, 'positions_short'] = -1
        self.df.loc[self.df.short_exit, 'positions_short'] = 0
        
        self.df.positions_short = self.df.positions_short.fillna(method='ffill')
        self.df['positions'] = self.df.positions_long + self.df.positions_short
        
        self.df['perc_change'] = ((self.df.spread - self.df.spread.shift(1))/(self.df[self.stock1] + self.model.params[0] * self.df[self.stock2])) * self.leverage
        self.df['strat_returns'] = self.df.positions.shift(1) * self.df.perc_change
        self.df['cum_returns'] = (self.df.strat_returns + 1).cumprod()
        
        print(f"Cummulative Returns are {100*(self.df['cum_returns'].iloc[-1]-1)}%")
        years = len(self.df) / 252
        self.cagr = (self.df['cum_returns'].iloc[-1]) ** (1/years) - 1
        self.annualized_stddev_returns = self.df['strat_returns'].std() * np.sqrt(252)
        profits = np.count_nonzero(self.df['strat_returns'] > 0)
        self.success_ratio = profits/len(self.df['strat_returns'])
        
        self.calc_drawdown()
        
        return (self.cagr, self.annualized_stddev_returns, self.success_ratio, np.max(self.df.drawdown), self.model.params[0], self.df)
        
    def calc_drawdown(self):
        max_rets = np.maximum.accumulate(self.df['cum_returns'])
        max_rets[max_rets < 1] = 1
        
        self.df.drawdown = self.df['cum_returns']/max_rets - 1
        