from urllib.request import Request, urlopen
import json
from datetime import datetime, timedelta
from iexfinance.stocks import get_historical_data
from iexfinance.refdata import get_symbols
import os
import numpy as np
import pandas as pd
#from numpy import random

def get_tickers_IEX():
    """
    Returns a dataframe with all the tickers available in IEX Cloud.
    token is stored in environment key IEX_TOKEN as sugested by IEX. 
    In this way, it is not necessary to call os.getenv() function
    """
    #iex_token = os.getenv("IEX_PUBLIC_KEY")
    
   # if type(iex_token) == str: print("IEX Key found successfully ...getting data")
   # else: return "Error: IEX Key NOT found"
    
    tickers=pd.DataFrame(get_symbols(output_format='pandas',
                                     #token=iex_token
                                    ))
    return tickers
    
def iex_search_ticker(look_x_tickers = [""],tickers_df=None):
    """
    Returns a dataframe with the tickers that matches the user input.
    Parameters: 
    tickers_df: a dataframe with the tickers. If it is not provided, 
    the function will get one automatically through get_tickers_IEX().
    look_x_tickers: a list with the tickers to look for. It can handle
    also a single string with the tickers separated by spaces.
    """
    
    #checks wether a dataframe is provided. If not, it is downloaded trhouhg get_tickers_IEX()
    if tickers_df is None:
        
            print(f"No database was provided \nGetting data base from IEX...")
            tickers_df=get_tickers_IEX()
           
            if tickers_df is not None: print("Succesfully downloaded database.")
            else: return "Data base could not be downloaded. Please try again later."
                
    #if a string is provided as "look_x_tickers" parameter, then it splits the string by 
    #spaces and store the outcome in a list.
    if  type(look_x_tickers) is str:
        look_x_tickers = look_x_tickers.split(" ")
        
    #checks if an invalid argument is passed trhough "look_x_tickers" parameter such
    #as no argument at all or an empty list '[]'. If that's the case, it will return a
    #message explaining why the method won't go forward after this point.
    if (len(look_x_tickers) == 1 and look_x_tickers[0] == "") or look_x_tickers==[]:
            return "Must provide at least one character to look for the stock ticker in databse"
               
    #If the parameters provided are valid, then it will return a dataframe "search_results".
    else:
        
        #create an empty dataframe to store the results later
        search_results = pd.DataFrame()
        #loops through the list of tickers to look for
        for stock in look_x_tickers:

            #stores the reults for each ticker and store that result in "search_results"
            results_for_ticker = tickers_df[tickers_df["symbol"].str.startswith(stock)]
            search_results = pd.concat([search_results,results_for_ticker], axis=0)
            
        return search_results
    
def select_tickers():
    """
    in construction
    """
    return "In construction"

####CHRISTIAN CODE######
def read_json(url):
    request = Request(url)
    response = urlopen(request)
    #print(response)
    data = response.read()
    #print(data)
    url2 = json.loads(data)
    return url2

def get_crypto_daily_price(cryptotickers = [], allData=False,limit = 90):
    """
    Returns a dataframe with the close prices of the cryptocurrencies selected. 
    Arguments:
    cryptotickers: list of tickers for currencies.
    allData: if True, gets all historical data available and ignores argument "limit".
    By default it is False.
    limit: the days from now to get the historical data. By default it's 90.
    """
    api_key = os.getenv("CC_API")
    ticker_list = cryptotickers
    crypto_df = pd.DataFrame()

    for ticker in ticker_list:
        #if allData is true, then it gets all the data available. If not, select data according to limit.
        if allData:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&allData=true&api_key={api_key}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&limit={limit}&api_key={api_key}"
       
        raw_data = read_json(url)
        #print(json.dumps(raw_data, indent=5))
        df = pd.DataFrame(raw_data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'],unit='s')
        df.set_index(df['time'], inplace=True)
        df['close'] = df['close'].astype(float)
        crypto_df[ticker] = df['close']
    
    #
    new_columns = pd.MultiIndex.from_product([ crypto_df.columns, ["close"]  ])
    crypto_df.columns = new_columns

    return crypto_df
    
### END CRHISTIAN CODE######

def normalize_dataframe(df=None):
    """
    Returns a dataframe with the normalize format for this library.
    The normalize format is a dataframe with only the close values of the ticker with the 
    columns named as the ticker. 
    Accepts dataframes with columns that are 2-level indexed where the upper index contains
    the ticker and the lower has at least one column named "close".
    """
    #Drops all the columns that are not 'close'
    col_to_drop = df.columns.levels[1].values
    col_to_drop = np.delete(col_to_drop,np.where(col_to_drop=="close"))
    df = df.drop(columns=col_to_drop, level=1)
    #eliminates the double level in columns by deleting the 'close' label. 
    #Only ticker label is necessary.
    df.columns = df.columns.droplevel(1)
    return df

def get_historic_data(end_date = datetime.now(), 
                      start_date = datetime.now() + timedelta(-365),
                      ticker=[],
                      close_only=True):
    """
    Returns a data frame with the HOLC and Volume info from the for "ticker" 
    provided list. The info is provided by IEX Cloud.
    
    Parameters:
    end_date: The final date for the historic data. By default, it's today.
    start_date: The starting date for the historic data. By default, it's today a year ago.
    Example: datetime.now() + timedelta(-365). "This is a year ago".
    ticker: the list of the tickers of the stocks that is attemped to get the historic data.
    e.g. ticker =["AAPL","GOOG","SQ"]
    Also accepts a single string with the the list of tickers separated by spaces. 
    e.g. "AAPL GOOG SQ"
    close_only: Boolean. If False, it will return HOLC and volume values. If True, it will 
    return only closing and volume data for the tickers. By default, it is True for efficiency.
    """
    #checks if the parameters provided through "ticker" is not an empty list
    #if it is, the function won't go forward after this point. returns explanatory message.
    if ticker == []:
        return "Empty list of tickers"
    
    #if a string is provided as "ticker" parameter, then it splits the string by 
    #spaces and store the outcome in a list.
    elif type(ticker) is str:
        ticker = ticker.split(" ")
            
    #iex_token = os.getenv("IEX_PUBLIC_KEY")#not necessary anymore.
    
    #Gets historical data with the parameters provided.
    #Gets only "close" and "volume" value for efficiency.
    prices = get_historical_data(ticker, start_date, end_date,
                                 output_format='pandas', 
                                 #token=iex_token, 
                                 close_only=close_only
                                )
    
    #If only one ticker is provided, then it adds another indexing level to the column
    #with the ticker. This is done for two reasons: 1) To visualize the ticker downloaded  
    #as a confirmation that I am working with correct data. 2) To mimic the format of the
    #dataframe obtained when getting 2 or more tickers data (2-level column indexing).
    if len(ticker) == 1:
        new_columns = pd.MultiIndex.from_product([ [ticker[0]],prices.columns ] )
        prices.columns = new_columns
        
    return prices

def get_crypto_olhc(crypto_ticker, allData=False,limit = 90):
    """
    Returns a dataframe with all features needed for ta lib see
    https://technical-analysis-library-in-python.readthedocs.io/en/latest/ for more information about ta for python. 
    Arguments:
    cryptoticker: Crypto ticker in string format.
    allData: if True, gets all historical data available and ignores argument "limit".
    By default it is False.
    limit: the days from now to get the historical data. By default it's 90.
    """
    api_key = os.getenv("CC_API")

    if allData:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={crypto_ticker}&tsym=USD&allData=true&api_key={api_key}"
    else:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={crypto_ticker}&tsym=USD&limit={limit}&api_key={api_key}"

    raw_data = mc.read_json(url)
    crypto_df = pd.DataFrame(raw_data['Data']['Data'])
    crypto_df['time'] = pd.to_datetime(crypto_df['time'],unit='s')
    ta_df = add_all_ta_features(crypto_df, "open", "high", "low", "close", "volumefrom", fillna=True)      

    return ta_df