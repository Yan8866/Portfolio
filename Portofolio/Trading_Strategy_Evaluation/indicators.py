import pandas as pd

def perc_B(df=None, windows=14):
    symbol = df.columns[0]
    # calculate SMA and rolling standard deviation in the window
    df['sma'] = df[symbol].rolling(window=windows).mean()
    df['rstd'] = df[symbol].rolling(window=windows).std()

    # calculate upper band, lower band and percent B
    df['upper_band'] = df['sma'] + df['rstd'] * 2
    df['lower_band'] = df['sma'] - df['rstd'] * 2
    df['percent_B'] = (df.iloc[:, 0] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
    percB = df['percent_B']
    return percB

def ppo(df = None, short=14, long=35, signal_line=9):

    # calculate ppo with 12-day EMA and 26-day EMA
    ppo_df = pd.DataFrame(index=df.index)
    ppo_df['short_EMA'] = df.ewm(span=short, adjust=False).mean()
    ppo_df['long_EMA'] = df.ewm(span=long, adjust=False).mean()
    ppo_df['ppo'] = ((ppo_df['short_EMA'] - ppo_df['long_EMA']) / ppo_df['long_EMA']) * 100

    # calculate signal line
    signal_line = ppo_df['ppo'].ewm(span=signal_line, adjust=False).mean()
    signal_line_df = pd.DataFrame(index=df.index)
    signal_line_df = signal_line_df.join(signal_line)
    signal_line_df = signal_line_df.rename(columns={'ppo': 'signal_line'})

    # calculate PPO Histogram
    ppo_hist = ppo_df['ppo'] - signal_line
    ppo_hist_df = pd.DataFrame(index=df.index)
    ppo_hist_df = ppo_hist_df.join(ppo_hist)
    ppo_hist_df = ppo_hist_df.rename(columns={'ppo': 'hist'})

    # join the three dataframes to identify signal line crossovers
    df_all = ppo_df.join(signal_line_df)
    df_all = df_all.join(ppo_hist_df)
    df_all['ratio'] = df_all['ppo']/ df_all['signal_line']
    ppo_ratio = df_all['ratio']
    return ppo_ratio

def stoc_osc(df=None, period=14, smoothing=3):

    # Create the "L14" column in the DataFrame
    df['Rolling_L'] = df['Low'].rolling(window=period).min()
    # Create the "H14" column in the DataFrame
    df['Rolling_H'] = df['High'].rolling(window=period).max()

    # Create the "%K" column in the DataFrame
    df['%K'] = 100 * ((df['Close'] - df['Rolling_L']) / (df['Rolling_H'] - df['Rolling_L']))
    # Create the "%D" column in the DataFrame
    df['%D'] = df['%K'].rolling(window=smoothing).mean()
    return df['%D']

def author():
    """
    #:return: The GT username of the student
    #:rtype: str
    """
    return "ycheng456"










