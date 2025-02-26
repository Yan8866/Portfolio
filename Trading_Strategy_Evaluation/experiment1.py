import datetime as dt
import pandas as pd
import math
import util as ut
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
import StrategyLearner as sl


start_date = dt.datetime(2008, 1, 2)
end_date = dt.datetime(2009, 12, 31)
symbols = ['JPM']
prices = ut.get_data(symbols, pd.date_range(start_date, end_date), addSPY=False)
prices = prices.dropna()

# build a new dataframe trades_bm
bm_dict = {'Date': prices.index[0],'Symbol':'JPM','Shares':1000}
bm_series = pd.Series(bm_dict)
trades_bm = pd.DataFrame([bm_series])
trades_bm.set_index('Date', inplace=True)
rv_bm = compute_portvals(trades=trades_bm,start_val=100000, commission=9.95, impact=0.005, sd=start_date, ed=end_date)
normed_rv_bm = rv_bm/rv_bm.iloc[0]

mystrategy = ManualStrategy()
df_trades = mystrategy.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
long_ms = df_trades[(df_trades['Shares'] == 2000)|(df_trades['Shares'] == 1000 )].index
short_ms = df_trades[df_trades['Shares'] == -2000].index
df_trades['Symbol'] = [symbols[0]] * len(df_trades)

rv_ms = compute_portvals(trades=df_trades,start_val=100000, commission=9.95, impact=0.005,sd=start_date, ed=end_date)
normed_rv_ms = rv_ms/rv_ms.iloc[0]

def port_stats(port_value=None, sample_freq = 252, risk_free_rate = 0):
    daily_rets= port_value/ port_value.shift(1) - 1
    daily_rets= daily_rets.iloc[1:]
    cr = port_value.iloc[-1] / port_value.iloc[0] - 1
    formatted_cr = cr.round(6)
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = math.sqrt(sample_freq) * (adr - risk_free_rate) / sddr
    return formatted_cr, adr, sddr, sr

cr_ms, adr_ms, sddr_ms, sr_ms = port_stats(port_value=rv_ms, sample_freq=252, risk_free_rate=0)
cr_bm, adr_bm, sddr_bm, sr_bm = port_stats(port_value=rv_bm, sample_freq=252, risk_free_rate=0)

start_date_t = dt.datetime(2010, 1, 2)
end_date_t = dt.datetime(2011, 12, 31)
prices_t = ut.get_data(symbols, pd.date_range(start_date_t, end_date_t), addSPY=False)
prices_t = prices_t.dropna()
prices_tcopy = prices_t.copy()

 # build a new dataframe trades_bm_t for out of sample data
bm_dict_t = {'Date': prices_t.index[0], 'Symbol': 'JPM', 'Shares': 1000}
bm_series_t = pd.Series(bm_dict_t)
trades_bm_t = pd.DataFrame([bm_series_t])
trades_bm_t.set_index('Date', inplace=True)
rv_bm_t = compute_portvals(trades=trades_bm_t, start_val=100000, commission=9.95, impact=0.005,
                              sd=start_date_t, ed=end_date_t)
normed_rv_bm_t = rv_bm_t / rv_bm_t.iloc[0]

mystrategy = ManualStrategy()
df_trades_t = mystrategy.testPolicy(symbol='JPM', sd=start_date_t, ed=end_date_t, sv=100000)
long_ms_t = df_trades_t[(df_trades_t['Shares'] == 2000) |(df_trades_t['Shares'] == 1000)].index
short_ms_t = df_trades_t[df_trades_t['Shares'] == -2000].index
df_trades_t['Symbol'] = [symbols[0]] * len(df_trades_t)
rv_ms_t = compute_portvals(trades=df_trades_t, start_val=100000, commission=9.95, impact=0.005,
                              sd=start_date_t, ed=end_date_t)
normed_rv_ms_t = rv_ms_t / rv_ms_t.iloc[0]

cr_ms_t, adr_ms_t, sddr_ms_t, sr_ms_t = port_stats(port_value=rv_ms_t, sample_freq=252, risk_free_rate=0)
cr_bm_t, adr_bm_t, sddr_bm_t, sr_bm_t = port_stats(port_value=rv_bm_t, sample_freq=252, risk_free_rate=0)

learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
learner.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                        sv=100000)  # training phase
df_trades_sl = learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                       sv=100000)  # testing phase
df_trades_sl['Symbol'] = [symbols[0]] * len(df_trades_sl)

rv_sl = compute_portvals(trades=df_trades_sl, start_val=100000, commission=9.95, impact=0.005, sd=start_date,
                              ed=end_date)
normed_rv_sl = rv_sl / rv_sl.iloc[0]

cr_sl, adr_sl, sddr_sl, sr_sl = port_stats(port_value=rv_sl, sample_freq=252, risk_free_rate=0)

# plot the normalized portfolio values
date = prices.index
fig = plt.figure()
fig.set_size_inches(7, 5)
plt.plot(date, normed_rv_bm, color='tab:purple', label="Benchmark")
plt.plot(date, normed_rv_ms, 'r-', label="MS")
plt.plot(date, normed_rv_sl, 'b-', label="SL")
plt.title('StrategyLearner vs Benchmark-In Sample')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.xticks(rotation=30)
plt.legend()
plt.savefig("./images/bm_ms_sl.png")
plt.clf()


# compute the out of sample portfolio value based on Strategy Learner
learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
learner.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                        sv=100000)  # training phase
df_trades_sl_t = learner.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                  sv=100000)  # testing phase
df_trades_sl_t['Symbol'] = [symbols[0]] * len(df_trades_sl_t)
rv_sl_t = compute_portvals(trades=df_trades_sl_t, start_val=100000, commission=9.95, impact=0.005, sd=start_date_t,
                            ed=end_date_t)
normed_rv_sl_t = rv_sl_t / rv_sl_t.iloc[0]

cr_sl_t, adr_sl_t, sddr_sl_t, sr_sl_t = port_stats(port_value=rv_sl_t, sample_freq=252, risk_free_rate=0)

# plot the normalized portfolio values
date_t = prices_t.index
fig = plt.figure()
fig.set_size_inches(7, 5)
plt.plot(date_t, normed_rv_bm_t, color='tab:purple', label="Benchmark")
plt.plot(date_t, normed_rv_ms_t, 'r-', label="MS")
plt.plot(date_t, normed_rv_sl_t, 'b-', label="SL")
plt.title('StrategyLearner vs Benchmark - Out of Sample')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.xticks(rotation=30)
plt.legend()
plt.savefig("./images/bm_sl_t.png")
plt.clf()


with open("./p8_results.txt", "w") as out:
    out.write("In Sample Statistics: ")
    out.write("Cumulative Return of the Benchmark: " + str(cr_bm) + "\n")
    out.write("Mean of Daily Returns of the Benchmark: " + str(adr_bm) + "\n")
    out.write("Stdev of Daily Returns of the Benchmark: " + str(sddr_bm) + "\n")
    out.write("Sharpe Ratio of the Benchmark: " + str(sr_bm) + "\n")
    out.write("Out of Sample Statistics: ")
    out.write("Cumulative Return of the Benchmark: " + str(cr_bm_t) + "\n")
    out.write("Mean of Daily Returns of the Benchmark: " + str(adr_bm_t) + "\n")
    out.write("Stdev of Daily Returns of the Benchmark: " + str(sddr_bm_t) + "\n")
    out.write("Sharpe Ratio of the Benchmark: " + str(sr_bm_t) + "\n")

    out.write("Manual Strategy Statistics: ")
    out.write("In Sample Cumulative Return of the MS: " + str(cr_ms) + "\n")
    out.write("In Sample Mean of Daily Returns of MS: " + str(adr_ms) + "\n")
    out.write("In Sample Stdev of Daily Returns of MS: " + str(sddr_ms) + "\n")
    out.write("In Sample Sharpe Ratio of MS: " + str(sr_ms) + "\n")
    out.write("Out of Sample Cumulative Return of MS: " + str(cr_ms_t) + "\n")
    out.write("Out of Sample Mean of Daily Returns of MS: " + str(adr_ms_t) + "\n")
    out.write("Out of Sample Stdev of Daily Returns of MS: " + str(sddr_ms_t) + "\n")
    out.write("Out of Sample Sharpe Ratio of MS: " + str(sr_ms_t) + "\n")

    out.write("Strategy Learner Statistics: ")
    out.write("In Sample Cumulative Return of SL: " + str(cr_sl) + "\n")
    out.write("In Sample Mean of Daily Returns of SL: " + str(adr_sl) + "\n")
    out.write("In Sample Stdev of Daily Returns of SL: " + str(sddr_sl) + "\n")
    out.write("In Sample Sharpe Ratio of SL: " + str(sr_sl) + "\n")
    out.write("Out of Sample Cumulative Return of SL: " + str(cr_sl_t) + "\n")
    out.write("Out of Sample Mean of Daily Returns of SL: " + str(adr_sl_t) + "\n")
    out.write("Out of Sample Stdev of Daily Returns of SL: " + str(sddr_sl_t) + "\n")
    out.write("Out of Sample Sharpe Ratio of SL: " + str(sr_sl_t) + "\n")

def author():
   """
   #:return: The GT username of the student
   #:rtype: str
   """
   return "ycheng456"
