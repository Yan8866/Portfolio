import datetime as dt
import pandas as pd
import util as ut
import math
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
import StrategyLearner as sl

impact1 = 0.001
impact2 = 0.01
impact3 = 0.05
impact4 = 0.1

start_date = dt.datetime(2008, 1, 2)
end_date = dt.datetime(2009, 12, 31)
symbols = ['JPM']
prices = ut.get_data(symbols, pd.date_range(start_date, end_date), addSPY=False)
prices = prices.dropna()


def port_stats(port_value=None, sample_freq=252, risk_free_rate=0):
    daily_rets = port_value / port_value.shift(1) - 1
    daily_rets = daily_rets.iloc[1:]
    cr = port_value.iloc[-1] / port_value.iloc[0] - 1
    formatted_cr = cr.round(6)
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = math.sqrt(sample_freq) * (adr - risk_free_rate) / sddr
    return formatted_cr, sddr,sr

#calculate portfolio statistics for impact1
learner1 = sl.StrategyLearner(verbose=False, impact=impact1, commission=0.00)  # constructor
learner1.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                        sv=100000)  # training phase
df_trades_sl1 = learner1.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                       sv=100000)

df_trades_sl1['Symbol'] = [symbols[0]] * len(df_trades_sl1)
rv_sl1 = compute_portvals(trades=df_trades_sl1, start_val=100000, commission=0.00, impact=impact1, sd=start_date,
                              ed=end_date)
normed_rv_sl1 = rv_sl1/rv_sl1.iloc[0]
cr_sl1, sddr_sl1, sr_sl1 = port_stats(port_value=rv_sl1, sample_freq=252, risk_free_rate=0)

#calculate portfolio statistics for impact2
learner2 = sl.StrategyLearner(verbose=False, impact=impact2, commission=0.00)  # constructor
learner2.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                        sv=100000)  # training phase
df_trades_sl2 = learner2.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                       sv=100000)

df_trades_sl2['Symbol'] = [symbols[0]] * len(df_trades_sl2)
rv_sl2 = compute_portvals(trades=df_trades_sl2, start_val=100000, commission=0.00, impact=impact2, sd=start_date,
                              ed=end_date)
normed_rv_sl2 = rv_sl2/rv_sl2.iloc[0]
cr_sl2, sddr_sl2, sr_sl2 = port_stats(port_value=rv_sl2, sample_freq=252, risk_free_rate=0)

#calculate portfolio statistics for impact3
learner3 = sl.StrategyLearner(verbose=False, impact=impact3, commission=0.00)  # constructor
learner3.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                        sv=100000)  # training phase
df_trades_sl3 = learner3.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                       sv=100000)
df_trades_sl3['Symbol'] = [symbols[0]] * len(df_trades_sl3)
rv_sl3 = compute_portvals(trades=df_trades_sl3, start_val=100000, commission=0.00, impact=impact3, sd=start_date,
                              ed=end_date)
normed_rv_sl3 = rv_sl3/rv_sl3.iloc[0]
cr_sl3, sddr_sl3, sr_sl3= port_stats(port_value=rv_sl3, sample_freq=252, risk_free_rate=0)

#calculate portfolio statistics for impact4
learner4 = sl.StrategyLearner(verbose=False, impact=impact4, commission=0.00)  # constructor
learner4.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                        sv=100000)  # training phase
df_trades_sl4 = learner4.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                       sv=100000)
df_trades_sl4['Symbol'] = [symbols[0]] * len(df_trades_sl4)
rv_sl4 = compute_portvals(trades=df_trades_sl4, start_val=100000, commission=0.00, impact=impact3, sd=start_date,
                              ed=end_date)
normed_rv_sl4 = rv_sl4/rv_sl4.iloc[0]

cr_sl4, sddr_sl4, sr_sl4= port_stats(port_value=rv_sl4, sample_freq=252, risk_free_rate=0)

# plot the normalized portfolio values with different impacts
date = prices.index
fig = plt.figure()
fig.set_size_inches(7, 5)
plt.plot(date, normed_rv_sl1, color='tab:purple', label="Impact=0.001")
plt.plot(date, normed_rv_sl2, 'b-', label="Impact=0.01")
plt.plot(date, normed_rv_sl3, 'g-', label="Impact=0.05")
plt.plot(date, normed_rv_sl4, 'r-', label="Impact=0.1")
plt.title('Portfolio Values with Different Impacts - In Sample')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.xticks(rotation=30)
plt.legend()
plt.savefig("./images/PortValue-Impacts.png")
plt.clf()

with open("./p8_results.txt", "a") as out:
    out.write("Statistics with Impact = 0.001: ")
    out.write("Cumulative Return: " + str(cr_sl1) + "\n")
    out.write("STDEV of Daily Returns: " + str(sddr_sl1) + "\n")
    out.write("Sharpe Ratio: " + str(sr_sl1) + "\n")

    out.write("Statistics with Impact = 0.01: ")
    out.write("Cumulative Return: " + str(cr_sl2) + "\n")
    out.write("STDEV of Daily Returns: " + str(sddr_sl2) + "\n")
    out.write("Sharpe Ratio: " + str(sr_sl2) + "\n")

    out.write("Statistics with Impact = 0.05: ")
    out.write("Cumulative Return: " + str(cr_sl3) + "\n")
    out.write("STDEV  of Daily Returns: " + str(sddr_sl3) + "\n")
    out.write("Sharpe Ratio: " + str(sr_sl3) + "\n")

    out.write("Statistics with Impact = 0.1: ")
    out.write("Cumulative Return: " + str(cr_sl4) + "\n")
    out.write("STDEV of Daily Returns: " + str(sddr_sl4) + "\n")
    out.write("Sharpe Ratio: " + str(sr_sl4) + "\n")

def author():
   """
   #:return: The GT username of the student
   #:rtype: str
   """
   return "ycheng456"