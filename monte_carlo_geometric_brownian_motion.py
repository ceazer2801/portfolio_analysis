def mcs_brownian (df, num_simulations):
# Monte Carlo Simulation with Geometric Brownian Motion
# simulated_price = previous_day_price * exp((daily_returns_mean - ((daily_std_mean**2)/2)) + (daily_std_mean * random_noise ))
    
    daily_returns_mean = df.mean()['close']
    daily_std_mean = df.std()['close']
    random_noise = np.random.normal()
    
    simulations = num_simulations
    trading_days = 252
    df_last_price = df['close'][-1]
    
    
    simulated_price_df = pd.DataFrame()
    portfolio_cum_returns = pd.DataFrame()
    
    for n in range(num_simulations):
        
        simulated_prices = [df_last_price]
        
        for i in range (trading_days):
            
            simulated_price = simulated_prices[-1] * np.exp((daily_returns_mean - ((daily_std_mean ** 2) / 2)) + (daily_std_mean * random_noise))
            simulated_prices.append(simulated_price)
            
        simulated_price_df = pd.Series(simulated_prices)
        
    return simulated_price_df