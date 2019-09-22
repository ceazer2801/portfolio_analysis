#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import panel as pn
from panel.interact import interact
import libs.dashboard as dashboard
import libs.apis as api
import matplotlib.pyplot as plt
import plotly.express as px
import panel as pn
#get_ipython().run_line_magic('matplotlib', 'inline')     <--- Christian is having an error with this, commented out to see if it breaks.
pn.extension("plotly")


# In[26]:
def bb_plot(df):
    bb_plot = plt.figure(figsize=(12,8));
    plt.plot(df.close)
    plt.plot(df.volatility_bbh, label='High BB')
    plt.plot(df.volatility_bbl, label='Low BB')
    plt.plot(df.volatility_bbm, label='EMA BB')
    plt.title('Bollinger Bands')
    plt.legend()
    plt.close()
    return pn.Pane(bb_plot)

def ichi_plot(df):
    ichi_plot = plt.figure(figsize=(12,8));
    plt.plot(df.close)
    plt.plot(df.trend_ichimoku_a, label='Ichimoku a')
    plt.plot(df.trend_ichimoku_b, label='Ichimoku b')
    plt.title('Ichimoku Kinko Hyo')
    plt.legend()
    plt.close()
    return pn.Pane(ichi_plot)

def macd_plot(df):
    macd_plot = plt.figure(figsize=(12,8));
    plt.plot(df.trend_macd, label='MACD');
    plt.plot(df.trend_macd_signal, label='MACD Signal')
    plt.plot(df.trend_macd_diff, label='MACD Difference')
    plt.title('MACD, MACD Signal and MACD Difference')
    plt.legend()
    plt.close()
    return pn.Pane(macd_plot)

def ema_plot(df):
    ema_plot = plt.figure(figsize=(12,8));
    plt.plot(df.close)
    plt.plot(df.volatility_bbm, label='EMA BB')
    plt.title('Exponential Moving Average')
    plt.legend()
    plt.close()
    return pn.Pane(ema_plot)

def get_ta(ta_df):
    """
    ema = ema_plot(ta_df)
    bb = bb_plot(ta_df)
    macd = macd_plot(ta_df)
    ichi= ichi_plot(ta_df)
    """
    
    def get_ta_child(indicator):
    
        if indicator == "Exponential Moving Avarage (EMA)": return ema_plot(ta_df)

        elif indicator == "Bolinger Bands": return bb_plot(ta_df)

        elif indicator == "MACD": return macd_plot(ta_df)

        elif indicator == "Ichimoku Kinkō Hyō.. Whaaaat??": return ichi_plot(ta_df)
        
    
    ta_options = ["Exponential Moving Avarage (EMA)","Bolinger Bands","MACD",
              "Ichimoku Kinkō Hyō.. Whaaaat??"]
    
    return interact(get_ta_child,indicator = ta_options)
    
    

    

def ta_pane(asset):
    
    ta_df = api.get_crypto_olhc(asset, allData=False, limit=30) #improve later for IEX options too
    print(ta_df.head(2))
    
    
    ta_pane = get_ta(ta_df)
    return ta_pane



crypto_checkboxes = pn.widgets.CheckBoxGroup(name='Cryptocurrencies', value=['BTC'], 
                                  options=["BTC","XRP","ETH","LTC","BCH","XLM"],inline=True)

index_checkboxes = pn.widgets.CheckBoxGroup(name='Index', value=['VOO'], 
                                  options=["VOO","VXF","VEA","BSV","BNDX","FLRN"],inline=True)

crypto_row_upper = pn.Row(crypto_checkboxes)
crypto_row_lower = pn.Row(index_checkboxes)
select_button = pn.widgets.Button(name="Select these assets", button_type='primary')

def click_select_button_evnt(event):
    ticker_dict = {"crypto": crypto_checkboxes.value,
              "index": index_checkboxes.value}
    
    panel  = dashboard.get_dashboard(ticker_dict, mc_trials = 100)
    panel.show()
    
select_button.on_click(click_select_button_evnt)

assets=["BTC","XRP","ETH","LTC","BCH","XLM"]

crypto_rows_column = pn.Column(interact(ta_pane, asset = assets ),crypto_row_upper, crypto_row_lower, select_button, 
                              align = "center", margin= (25,25))

crypto_rows_column.show()


# In[24]:




