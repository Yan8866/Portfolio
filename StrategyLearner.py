
import datetime as dt
import pandas as pd
import numpy as np
import util as ut
from indicators import perc_B, ppo, stoc_osc
import RTLearner as rtl
import BagLearner as bl
class StrategyLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 

    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact = 0.0, commission=0.0):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission
        self.learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={'leaf_size':5}, bags=40, boost=False, verbose=False)

    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 
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

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd - dt.timedelta(days=50), ed + dt.timedelta(days=50))
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
        prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 

        prices_1 = prices.copy()
        prices_2 = prices.copy()
        prices_copy1 = prices.copy()
        prices_copy2 = prices.copy()

        # get indicators
        percB = perc_B(df=prices_copy1, windows=15)
        std_percB = (percB - percB.mean())/percB.std()
        ind_ppo = ppo(df=prices_copy2, short=10, long=30, signal_line=6)
        std_ppo = (ind_ppo - ind_ppo.mean()) / ind_ppo.std()

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

        ind_stocha = stoc_osc(df=prices_df, period=14, smoothing=3)
        std_ind_stocha = (ind_stocha - ind_stocha.mean()) / ind_stocha.std()

        #contruct x_train
        actual_dates = prices.index[(prices.index >= sd) & (prices.index <= ed)]
        x_train = pd.DataFrame({'bbp': std_percB, 'ppo': std_ppo,'stoc': std_ind_stocha}, index=actual_dates)

        # construct y_train
        y_train = pd.DataFrame(0, index=actual_dates, columns=syms)
        prices_train = prices_2.loc[prices_2.index.isin(actual_dates)]
        YBUY = 0.038
        YSELL = -0.04
        n = len(actual_dates)
        ret_buy = np.zeros(n)
        ret_sell = np.zeros(n)
        for i in range(n-8):
            ret_buy[i] = float((prices_train.iloc[i+8,:] * (1 +self.impact)) / prices_train.iloc[i,:] - 1)
            ret_sell[i] = float((prices_train.iloc[i+8, :] * (1-self.impact)) / prices_train.iloc[i, :] - 1)
            if ret_buy[i] > YBUY:
                y_train.values[i] = 1
            elif ret_sell[i] < YSELL:
                y_train.values[i] = -1

        self.learner.add_evidence(x_train.to_numpy(), y_train.to_numpy())

        if self.verbose:
            print(prices)
  		  	   		 	   			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	   			  		 			     			  	 
        sv=10000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
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
        prices = ut.get_data(syms, dates, False)
        prices = prices.dropna()

        prices_1 = prices.copy()
        prices_copy1 = prices.copy()
        prices_copy2 = prices.copy()

        # get indicators
        percB = perc_B(df=prices_copy1, windows=15)
        std_percB = (percB - percB.mean())/percB.std()
        ind_ppo = ppo(df=prices_copy2, short=10, long=30, signal_line=6)
        std_ppo = (ind_ppo - ind_ppo.mean()) / ind_ppo.std()

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

        ind_stocha = stoc_osc(df=prices_df, period=14, smoothing=3)
        std_ind_stocha = (ind_stocha - ind_stocha.mean()) / ind_stocha.std()

        # contruct x_test
        actual_dates = prices.index[(prices.index >= sd) & (prices.index <= ed)]
        x_test = pd.DataFrame({'bbp': std_percB, 'ppo': std_ppo, 'stoc': std_ind_stocha},
                               index=actual_dates)

        # construct y_test
        result = self.learner.query(x_test.to_numpy())
        y_test = result.mode
        ytest_df = pd.DataFrame(y_test, index=actual_dates, columns=['action'])

        def trade_orders(self, raw_signals):
            # Create an orders dataframe
            orders = pd.DataFrame(index=raw_signals.index.values, columns=["Shares"])
            for i in range(len(orders.index) - 1):
                if raw_signals['action'][i] == 1:
                    if orders['Shares'].sum() == 0:
                        orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 1000
                    elif orders['Shares'].sum() == -1000:
                        orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 2000
                    elif orders['Shares'].sum() == 1000:
                        orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 0
                elif raw_signals['action'][i] == -1:
                    if orders['Shares'].sum() == 0:
                        orders.iloc[i + 1, orders.columns.get_loc('Shares')] = -1000
                    elif orders['Shares'].sum() == 1000:
                        orders.iloc[i + 1, orders.columns.get_loc('Shares')] = -2000
                    elif orders['Shares'].sum() == -1000:
                        orders.iloc[i + 1, orders.columns.get_loc('Shares')] = 0
            return orders

        df_trades = trade_orders(self, ytest_df)
        df_trades = df_trades.dropna()
        return df_trades

    def author():
        """
        #:return: The GT username of the student
        #:rtype: str
        """
        return "ycheng456"

if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	   			  		 			     			  	 
