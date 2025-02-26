import datetime as dt
import pandas as pd
import util as ut
import math
from marketsimcode import compute_portvals
import StrategyLearner as sl
import itertools

# import stock price data
start_date = dt.datetime(2008, 1, 2)
end_date = dt.datetime(2009, 12, 31)
symbols = ['JPM']
prices = ut.get_data(symbols, pd.date_range(start_date, end_date), addSPY=False)
prices = prices.dropna()

# Define parameter ranges
perc_B_windows_range = [10,15,20]
ppo_short_range = [10, 14]
ppo_long_range = [30, 50]
ppo_signal_line_range = [6, 9]
stoc_period_range = [14, 20]
stoc_smoothing_range = [3]
leaf_size_range = [5,6,7]
bags_range = [20,30, 40]
YBUY_range = [0.038]
YSELL_range = [-0.038, -0.04]
days_for_Y_range = [8, 9]

best_sr = float('-inf')
best_params = {}

def port_stats(port_value=None, sample_freq=252, risk_free_rate=0):
    daily_rets = port_value / port_value.shift(1) - 1
    daily_rets = daily_rets.iloc[1:]
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = math.sqrt(sample_freq) * (adr - risk_free_rate) / sddr
    return sr

# Iterate through parameter combinations
for (perc_B_windows, ppo_short, ppo_long, ppo_signal_line, stoc_period, stoc_smoothing, leaf_size, bags, YBUY, YSELL, days_for_Y) in itertools.product(perc_B_windows_range, ppo_short_range, ppo_long_range, ppo_signal_line_range, stoc_period_range,
        stoc_smoothing_range, leaf_size_range, bags_range, YBUY_range, YSELL_range, days_for_Y_range):

    # Train and test StrategyLearner with current parameters
    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                          sv=100000)
    df_trades_sl = learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                        sv=100000)
    df_trades_sl['Symbol'] = [symbols[0]] * len(df_trades_sl)
    port_vals = compute_portvals(trades=df_trades_sl, start_val=100000, commission=9.95, impact=0.005, sd=start_date,
                              ed=end_date)
    # Calculate Sharpe ratio
    sr = port_stats(port_value=port_vals)
    sr = sr.iloc[0]
    # Update best parameters if current Sharpe ratio is higher
    if sr > best_sr:
        best_sr = sr
        best_params = {
            'perc_B_windows': perc_B_windows,
            'ppo_short': ppo_short,
            'ppo_long': ppo_long,
            'ppo_signal_line': ppo_signal_line,
            'stoc_period': stoc_period,
            'stoc_smoothing': stoc_smoothing,
            'leaf_size': leaf_size,
            'bags': bags,
            'YBUY': YBUY,
            'YSELL': YSELL,
            'days_for_Y': days_for_Y}
        """
        if best_sr >= 2.0:
            print("Best parameters:", best_params)
            print("Best Sharpe ratio:", best_sr)
        """


def author():
    """
    #:return: The GT username of the student
    #:rtype: str
    """
    return "ycheng456"





