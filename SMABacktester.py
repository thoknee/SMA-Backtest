import pandas as pd
import numpy as np
from scipy.optimize import brute
import yfinance as yf
import matplotlib.pyplot as plt


class SMABacktester:
    def __init__(self, symbol, SMA1, SMA2, start, end=None):
    
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end

        # Sharpe ratios for buy/hold and strategy
        self.sharpeBH = None
        self.sharpe = None
        
        self.results = None
    
        self.get_data()
    
    
    def get_data(self):
        # Downloads yfinance data
        if self.end != None:
            data = yf.download(self.symbol, self.start, self.end)
        else:
            data = yf.download(self.symbol, self.start)
        
        # Calculates log returns and simple moving averages
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))

        data['SMA1'] = data['returns'].rolling(self.SMA1).mean()
        data['SMA2'] = data['returns'].rolling(self.SMA2).mean()

        self.data = data

    def run_strategy(self):
        # Get a copy of the data
        data = self.data.copy().dropna()

        # Calculate position based on golden/death cross then calculate returns 
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strat'] = data['position'].shift(1) * data['returns']

        # Cumulative returns of strategy and of just buying/holding asset
        data['stratReturns'] = data['strat'].cumsum().apply(np.exp)
        data['buyHold'] = data['returns'].cumsum().apply(np.exp)

        # Calculates annualize sharpe ratios
        self.sharpeBH = (data['returns'].mean()/data['returns'].std()) * np.sqrt(252)
        self.sharpe = (data['strat'].mean()/data['strat'].std()) * np.sqrt(252)
    
        self.results = data

        # Calculates performance of strategy and buy/hold
        # Calculates outperformance of strategy over buy/hold
        stratPerformance = data['stratReturns'].iloc[-1]
        buyHoldPerformance = data['returns'].iloc[-1]

        outPerformance = stratPerformance - buyHoldPerformance

        return round(stratPerformance, 2), round(outPerformance, 2)

    def visual(self, graph):
        # Visualizes data

        if graph.lower() == 'returns':
            title = '%s | SMA1 = %d, SMA2 = %d' % (self.symbol, self.SMA1, self.SMA2)
            self.results[['stratReturns', 'buyHold']].plot(title = title)
            
            print('Annualized buy and hold Sharpe: ', self.sharpeBH)
            print('Annualized strategy Sharpe: ', self.sharpe)
        
        if graph.lower() == 'sma':
            # Copy over data
            data = self.results.copy()
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(24,16))
            
            # Top panel: Price
            data['Close'].plot(ax=ax1, color='black', title="Price with Return-SMA Signals")
            ax1.set_ylabel("Price")
            
            # Mark crossovers on price
            cross_up = (data['SMA1'] > data['SMA2']) & (data['SMA1'].shift(1) <= data['SMA2'].shift(1))
            cross_dn = (data['SMA1'] < data['SMA2']) & (data['SMA1'].shift(1) >= data['SMA2'].shift(1))
            
            ax1.scatter(data.index[cross_up], data['Close'][cross_up], marker='^', color='green', s=60, label="Bullish crossover")
            ax1.scatter(data.index[cross_dn], data['Close'][cross_dn], marker='v', color='red', s=60, label="Bearish crossover")
            ax1.legend()
            
            # SMAs
            data[['SMA1', 'SMA2']].plot(ax=ax2)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax2.set_ylabel("SMA of Log Returns")


            
    # Find most optimal simple moving averages.
    def set_params(self, SMA1 = None, SMA2 = None):
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['returns'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['returns'].rolling(self.SMA2).mean()
            
    def optFunc(self, SMA):
        # Function we are trying to minimize (negative returns of strategy)
        self.set_params(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

        
    
    def optimize(self, SMA1, SMA2):
        # SMA1 and SMA2 must by tuples in the form ('minimum', 'maximum', 'step_size')
        opt = brute(self.optFunc, (SMA1, SMA2), finish = None)
        return opt, -self.optFunc(opt)
        