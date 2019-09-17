#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import hvplot.pandas
import montecarlo as mc
import seaborn as sns
import panel as pn
from panel.interact import interact

from iexfinance.stocks import get_historical_data
import iexfinance as iex


# In[2]:


tickers_dict = {"index":["VOO"],
                "crypto":['BTC', 'ETH', 'XRP']}
def get_assets_hist_data(tickers_dict={"index":[],"crypto":[]}, years=2):
    
    #Defining starting dat to get historical data.
    data_start_date = datetime.now() + timedelta(int(-365*years))

    #getting indeces historical prices form IEX
    portfolio_hist_prices = mc.get_historic_data(ticker = tickers_dict["index"], 
                                                 start_date = data_start_date)

    #getting cryptos historical prices form cryptocompare
    btc_daily_price = mc.get_crypto_daily_price(tickers_dict["crypto"],limit=int(years*365))

    #concatenating both dataframes
    portfolio_hist_prices = pd.concat([portfolio_hist_prices,btc_daily_price],axis=1,join="inner")
    portfolio_hist_prices.dropna(inplace=True)
    portfolio_hist_prices = portfolio_hist_prices[(portfolio_hist_prices[portfolio_hist_prices.columns] != 0).all(axis=1)]

    #formating dataframes
    portfolio_hist_prices = mc.normalize_dataframe(portfolio_hist_prices)
    portfolio_daily_retn = portfolio_hist_prices.pct_change().copy()
    
    #Save both hist. prices and hist. daily returns dataframes packed in a list to be able to return in the funtion.
    hist_price_ret_df = [ portfolio_hist_prices, portfolio_daily_retn ]
    
    return hist_price_ret_df


def corr_plot(portfolio_daily_retn):
    correlated = portfolio_daily_retn.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlated, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    correlated_plot, ax = plt.subplots(figsize=(6, 4))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlated, mask=mask, cmap="coolwarm", vmax=1, vmin =-1, 
                center=0,square=True, linewidths=.5, annot=True )

     #correlated_plot = sns.heatmap(correlated, vmin=-1, vmax=1, annot=True,cmap="coolwarm") 
    plt.close()
    return pn.Pane(correlated_plot)


def sharp_rt_plot(portfolio_daily_retn):

    sharp_ratios = portfolio_daily_retn.mean()*np.sqrt(252)/portfolio_daily_retn.std()

    sr_plot = plt.figure();
    plt.bar(x = sharp_ratios.index, height=sharp_ratios)
    plt.axhline(sharp_ratios.mean(), color='r')
    plt.close()
    return sr_plot


#monte_carlo_sim = mc.monte_carlo_sim(df=portfolio_hist_prices, trials=10, sim_days=252)

def plot_mont_carl(monte_carlo_sim):
    plot_title = f"title"
    monte_carlo_sim_plot = monte_carlo_sim.hvplot(title=plot_title,figsize=(18,10))
    return monte_carlo_sim_plot


def get_conf_interval(last_row_db,q=[0.05, 0.95]):
    confidence_interval = last_row_db.quantile(q=q)
    return confidence_interval


def plot_conf(values=None,conf=[0,0]):
    conifidence_plot = plt.figure();
    plt.hist(x = values,bins=20)
    plt.axvline(conf.iloc[0], color='r')
    plt.axvline(conf.iloc[1], color='r')
    plt.close()
    return pn.Pane(conifidence_plot)


def get_dashboard(tickers_dict={"index":[],"crypto":[]}, years=2, mc_trials=500, mc_sim_days=252):
    
    data = get_assets_hist_data(tickers_dict=tickers_dict, years=years)
    
    mc_sim = mc.monte_carlo_sim(data[0],trials = mc_trials, sim_days = mc_sim_days)
    
    risk_tabs = pn.Tabs(
        ("Correlation of portfolio",corr_plot(data[1])),
        ("Sharp Ratios", sharp_rt_plot(data[1]))
    )


    montecarlo_tabs = pn.Tabs(
        ("monte Carlo Simulation",plot_mont_carl(mc_sim)),
        ("Confidence Intervals", plot_conf(mc_sim.iloc[-1],get_conf_interval(mc_sim.iloc[-1])))
    )

    techl_analysis_tabs = pn.Tabs(
        ("TA1","in construction"),
        ("TA2", "in construction")
    )

    tabs = pn.Tabs(
        ("Risk",risk_tabs),
        ("Monte Carlo Simulation", montecarlo_tabs),
        ("Tecnical Analysis", techl_analysis_tabs),
        ("Report", "in construction")
    )

    panel = tabs

    return panel


# In[ ]:




