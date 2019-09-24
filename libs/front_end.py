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
from datetime import datetime, timedelta
#get_ipython().run_line_magic('matplotlib', 'inline')     <--- Christian is having an error with this, commented out to see if it breaks.
pn.extension("plotly")

####List of ticker categorized by cryptocurrencies and indices.
cryptocurrencies = ["BTC","XRP","ETH","LTC","BCH","XLM"]
indices = ["VOO","VXF","VEA","BSV","BNDX","FLRN"]

def price_only(df):
    priceonly_plot = plt.figure(figsize=(12,8));
    plt.plot(df.close, "go-" ,linewidth=2, markersize=5)
    plt.title('Price over the last 30 days')
    plt.xticks(rotation="vertical") 
    plt.grid(True)
    plt.legend()
    plt.close()
    return pn.Pane(priceonly_plot)


def bb_plot(df):
    bb_plot = plt.figure(figsize=(12,8));
    plt.plot(df.close, "go-" ,linewidth=2, markersize=5)
    plt.plot(df.volatility_bbh, "--", linewidth=2,label='High BB')
    plt.plot(df.volatility_bbl, "--", linewidth=2,label='Low BB')
    plt.plot(df.volatility_bbm, "--", linewidth=2,label='EMA BB')
    plt.title('Bollinger Bands')
    plt.xticks(rotation="vertical") 
    plt.grid(True)
    plt.legend()
    plt.close()
    return pn.Pane(bb_plot)


def macd_plot(df):
    macd_plot = plt.figure(figsize=(12,8));
    plt.plot(df.trend_macd,"--", linewidth=2,label='MACD');
    plt.plot(df.trend_macd_signal,"--",linewidth=2, label='MACD Signal')
    plt.plot(df.trend_macd_diff, "--",linewidth=2,label='MACD Difference')
    plt.title('MACD, MACD Signal and MACD Difference')
    plt.xticks(rotation="vertical") 
    plt.grid(True)
    plt.legend()
    plt.close()
    return pn.Pane(macd_plot)

def ema_plot(df):
    ema_plot = plt.figure(figsize=(12,8));
    plt.plot(df.close, "go-" ,linewidth=2, markersize=5)
    plt.plot(df.volatility_bbm, "--",linewidth=2, label='EMA BB')
    plt.title('Exponential Moving Average')
    plt.xticks(rotation="vertical") 
    plt.grid(True)
    plt.legend()
    plt.close()
    return pn.Pane(ema_plot)

def get_ta(ta_df):

    def get_ta_child(indicator):
    
        if indicator == "Exponential Moving Avarage (EMA)": return ema_plot(ta_df)

        elif indicator == "Bolinger Bands": return bb_plot(ta_df)

        elif indicator == "MACD": return macd_plot(ta_df)
        
        elif indicator == "None": return price_only(ta_df)
        
    
    ta_options = ["None","Exponential Moving Avarage (EMA)","Bolinger Bands","MACD"]
    
    return interact(get_ta_child,indicator = ta_options)

def ta_pane(asset):
    
    if asset in cryptocurrencies:
        ta_df = api.get_crypto_olhc(asset, allData=False, limit=30)
        ta_df.set_index("time", inplace = True)
        print(ta_df.head())
    else:
        ta_df = api.get_historic_data( start_date = datetime.now() + timedelta(-30), 
                                      ticker=[asset], close_only = False)
        ta_df.columns = ta_df.columns.droplevel(0)
        print(ta_df.head())
        
        
     
    ta_pane = get_ta(ta_df)
    return ta_pane   


crypto_checkboxes = pn.widgets.CheckButtonGroup(name='Cryptocurrencies', value=[cryptocurrencies[0]], 
                                  options= cryptocurrencies,inline=True)

index_checkboxes = pn.widgets.CheckButtonGroup(name='Index', value=[indices[0]], 
                                  options=indices, inline=True)


crypto_row_upper = pn.Column('''
<h3>Select any of the Crypto Curriencies listed below: </h3>
''', crypto_checkboxes
                            )
crypto_row_lower = pn.Column('''
<h3>Select any of the Stock or Bond Indexes listed below: </h3>
''', index_checkboxes
                            )

crypto_selector_row = pn.Row(crypto_row_upper, crypto_row_lower)

select_button = pn.widgets.Button(name="Select Any Combination of Stock Indexes, Bond Indexes, and Coins Above then PRESS HERE to Generate a Sample Portfolio", button_type='primary')

def click_select_button_evnt(event):
    ticker_dict = {"crypto": crypto_checkboxes.value,
              "index": index_checkboxes.value}
    
    panel  = dashboard.get_dashboard(ticker_dict, mc_trials = 100)
    panel.show()
    
select_button.on_click(click_select_button_evnt)

assets= cryptocurrencies + indices

## New code injection

marqu_txt = api.get_marquee_text()   

m_text = pn.panel( 
marqu_txt, 
align = "center"
)

side_text = pn.pane.Markdown(
'''
<style>

body {
    background-color: #FFFFFF;
}

mar {
  color: #000000;
  text-align: center;
  font-family: "Times New Roman", Times, serif;
  font-style: normal;
  font-size: 17px;
}

#leftbox {
    color: black;
}

bold{
    font-weight: bold;
    color: #993300;
    text-align: center;
    font-family: "Times New Roman", Times, serif;
    font-style: oblique;
    font-size: 24px;
    font-variant: small-caps;
}
p {
  color: #000000;
}

p1 {
  color: #006600;
  font-size: 17px;
}

h1 {
    font-size: 30px;
    font-variant: small-caps;
    font-weight: bold;
    font-family: Arial, Helvetica, sans-serif;
}

h2 {
  color: #000000;
  font-family: Arial, Helvetica, sans-serif;
}

h3 {
    color: #000000
    font-size: 16px;
    font-style: italic;
}

cr {
    font-size: 14px;
    font-style: italic;
    color: #33CCFF;
}
</style>
            
<div id="leftbox"> 
<h1>Welcome to The Melting Pot</h1>
</div>
---
<h2>Let Us Help You Create Your Financial Fondue of Freedom...</h2>
</br>
</br>
<p1>Recently the world has been taken by storm by Crypto Currencies and their incredible potential for profits, but does adding cryptos make sense or is it adding too much risk?
Here at the Melting Pot we set out to answer just this question and more!
</br>
</br>
By now most people have heard about Bitcoin, but did you know there are currently 2,379 coins listed on Coin Market Cap?  We handpicked several of the most popular coins by market cap and give you the opportunity to test their performance with or without traditional stock and bond indexes.  
</br>
</br>
<bold>Use the Technical Analysis feature on this page to help decide what to populate your portfolio with.</bold></p1>
</br>
</br><p>Not sure what Crypto Currencies are? Want to learn more? <a href='https://cointelegraph.com/bitcoin-for-beginners/what-are-cryptocurrencies', target="_blank">Click Here</a>
---
''',
        align= "center",
        width_policy = "max",
    )
    
lower_text = pn.pane.Markdown('''
<h2><bold>Please wait for simulations and analysis to run after initiating the request, results will appear in a new window.</bold></h2>
---
        ''',
                                  align= "center",
                                  width_policy = "max",
                                  margin=(0, 50),
                                 )

left_row = pn.Row(side_text, align="start")
middle_row = pn.Row(interact(ta_pane, asset = assets ), align = "end", width_policy="min")
both_row = pn.Row(left_row, middle_row)

crypto_rows_column = pn.Column(both_row, crypto_selector_row, select_button,lower_text,align="center", sizing_mode='stretch_both')

crypto_rows_column.show()



