""""""
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 

Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 

We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 

-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 

Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: ycheng456 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903955239 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""

import datetime as dt
from indicators import perc_B, ppo, stoc_osc
import pandas as pd
import numpy as np
import util as ut

class ManualStrategy(object):
    """
    A Manual strategy learner that can learn a trading policy using the indicators chose by certain rules.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        pass

    def trade_orders(self, raw_signals):
        # Create an orders dataframe
        orders = pd.DataFrame(index=raw_signals.index.values, columns=["Shares"])
        for i in range(len(orders.index) - 1):
            if raw_signals[i] == 1:
                if orders['Shares'].sum() == 0:
                    orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 1000
                elif orders['Shares'].sum() == -1000:
                    orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 2000
                elif orders['Shares'].sum() == 1000:
                    orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 0
            elif raw_signals[i] == -1:
                if orders['Shares'].sum() == 0:
                    orders.iloc[i + 1, orders.columns.get_loc('Shares')] = -1000
                elif orders['Shares'].sum() == 1000:
                    orders.iloc[i + 1, orders.columns.get_loc('Shares')] = -2000
                elif orders['Shares'].sum() == -1000:
                    orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 0
        return orders

    def testPolicy(self, symbol="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31),
                   sv=10000):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        syms = [symbol]
        dates = pd.date_range(sd - dt.timedelta(days=50), ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        prices = prices.dropna()
        prices_1 = prices.copy()
        prices_copy1 = prices.copy()
        prices_copy2 = prices.copy()

        percB = perc_B(df=prices_copy1, windows=40)
        prices['perc_B'] = np.where(percB <= 0, '1', np.where(percB >= 1.1, '-1', '0'))
        ind_ppo = ppo(df=prices_copy2, short=14, long=35, signal_line=9)
        prices['ppo'] = np.zeros(prices.shape[0])
        for i in range(prices.shape[0]-1):
            if (ind_ppo.iloc[i]>=1.01) & (ind_ppo.iloc[i]<=1.1) & (ind_ppo.iloc[i] > ind_ppo.iloc[i+1]):
                prices['ppo'].iloc[i] = -1
            elif (ind_ppo.iloc[i]>=1.01) & (ind_ppo.iloc[i]<=1.1) & (ind_ppo.iloc[i] < ind_ppo.iloc[i+1]):
                prices['ppo'].iloc[i] = 1
            elif (ind_ppo.iloc[i]>=0.9) & (ind_ppo.iloc[i]<=0.99) & (ind_ppo.iloc[i] < ind_ppo.iloc[i+1]):
                prices['ppo'].iloc[i] = 1
            elif (ind_ppo.iloc[i]>= 0.9) & (ind_ppo.iloc[i]<=0.99) & (ind_ppo.iloc[i] > ind_ppo.iloc[i+1]):
                prices['ppo'].iloc[i] = -1
        prices.fillna(0, inplace=True)

        prices_1.columns = ["Adj Close"]
        open_price = ut.get_data(syms, dates, addSPY=False, colname="Open")
        open_price = open_price.dropna()
        open_price.columns = ['Open']
        high = ut.get_data(syms, dates, addSPY=False, colname="High")
        high = high.dropna()
        high.columns = ['High']
        low = ut.get_data(syms, dates, addSPY=False, colname="Low")
        low = low.dropna()
        low.columns = ['Low']
        close = ut.get_data(syms, dates, addSPY=False, colname="Close")
        close = close.dropna()
        close.columns = ['Close']

        prices_df = prices_1.join(open_price)
        prices_df = prices_df.join(high)
        prices_df = prices_df.join(low)
        prices_df = prices_df.join(close)
        prices_df = prices_df.dropna()

        ind_stocha = stoc_osc(df=prices_df, period=25, smoothing=3)
        prices['stochastic'] = np.where(ind_stocha <= 20, '1', np.where(ind_stocha >= 90, '-1', '0'))
        prices['perc_B'] = prices['perc_B'].astype('int64')
        prices['stochastic'] = prices['stochastic'].astype('int64')
        # Create the "order" column based on conditions
        signal_list = ['perc_B', 'ppo', 'stochastic']
        prices['raw_signal'] = np.nan
        for ind in prices.index:
            if (prices.loc[ind, signal_list] == 1).sum() >= 2:
                prices.loc[ind, 'raw_signal'] = 1
            elif ((prices.loc[ind, signal_list] == -1).sum() >=1):
                prices.loc[ind, 'raw_signal'] = -1
            else:
                prices.loc[ind, 'raw_signal'] = 0

        prices = prices[(prices.index >= sd) & (prices.index <= ed)]
        raw_signals = prices['raw_signal']
        df_trades = self.trade_orders(raw_signals)
        df_trades = df_trades.fillna(0)
        if self.verbose:
            print(prices.tail())

        return df_trades


def author():
    """
    #:return: The GT username of the student
    #:rtype: str
    """
    return "ycheng456"


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
