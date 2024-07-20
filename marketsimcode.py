 	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt
import numpy as np  		  	   		 	   			  		 			     			  	 
import pandas as pd
from util import get_data, plot_data


def compute_portvals(  		  	   		 	   			  		 			     			  	 
    trades = None,
    start_val=1000000,  		  	   		 	   			  		 			     			  	 
    commission=9.95,  		  	   		 	   			  		 			     			  	 
    impact=0.005,
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31)):
    """  		  	   		 	   			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param trades: trades for each trading day 		  	   		 	   			  		 			     			  	 
    :type trades: dataframe 		  	   		 	   			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
    :type start_val: int  		  	   		 	   			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 	   			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	   			  		 			     			  	 
    # code should work correctly with either input

    orders = trades
    orders.reset_index(inplace=True)
    orders.rename(columns={'index': 'Date'}, inplace=True)
    symbols = orders['Symbol'].unique()

    start_date = sd
    end_date = ed
    prices = get_data(symbols, pd.date_range(start_date, end_date), addSPY=False)
    prices = prices.dropna()
    actual_dates = prices.index[(prices.index >= sd) & (prices.index <= ed)]
    # build a new dataframe trades_df
    new_columns = prices.columns
    trades_df = pd.DataFrame(index=actual_dates, columns=new_columns)
    num_rows = len(prices)
    trades_df["CASH"] = np.zeros(num_rows)
    trades_df = trades_df.fillna(0)

    # update trades dataframe according to orders
    for index, row in orders.iterrows():
        date = row["Date"]
        shares = row["Shares"]
        symb = row["Symbol"]
        stock_price = prices.loc[date, symb]
        trades_df.loc[date, symb] = trades_df.loc[date, symb] + shares
        if shares != 0:
            if shares > 0:
                stock_price = prices.loc[date, symb] * (1 + impact)
            elif shares < 0:
                stock_price = prices.loc[date, symb] * (1 - impact)
            trades_df.loc[date, "CASH"] = trades_df.loc[date, "CASH"] - stock_price * shares - commission
    # build a new dataframe holdings
    holdings = trades_df.copy()

    for i in range(1, len(holdings)):
        start_D = holdings.index[0]
        holdings.loc[start_D, "CASH"] = start_val + trades_df.loc[start_D, "CASH"]
        holdings.iloc[i] = holdings.iloc[i] + holdings.iloc[i - 1]
    # build a new dataframe values
    values = holdings.copy()
    for index, row in values.iterrows():
        for i in range(0, values.shape[1] - 1):
            col_name = values.columns[i]
            values.loc[index, col_name] *= prices.loc[index, col_name]

    values["port_value"] = values.sum(axis=1)
    portvals = values["port_value"]
    rv = portvals.to_frame(name="port_value")
    return rv

  		  	   		 	   			  		 			     			  	 
def author():
    """
    #:return: The GT username of the student
    #:rtype: str
    """
    return "ycheng456"



