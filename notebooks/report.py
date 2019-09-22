#set test variables
voo = "VOO"
btc = "Bitcoin"
portfolio_sharpe_ratio=1.5
voo_sharpe=1
btc_sharpe=2

#compare portfolio to voo and btc interchangeably
def higher_or_lower(portfolio, instrument):
    """Compare portfolio to VOO and BTC interchangeably"""
    #portfolio = portfolio_sharpe_ratio
    if portfolio > instrument:
        portfolio_vs_instrument = "higher"
    elif portfolio == instrument:
        portfolio_vs_instrument = "the same as"
    else:
        portfolio_vs_instrument = "lower"
    return portfolio_vs_instrument

f"""Your Sharpe Ratio is {portfolio_sharpe_ratio}.
 The risk adjusted retun of your portfolio is {higher_or_lower(portfolio_sharpe_ratio, voo_sharpe)} than that of {voo} ({voo_sharpe}) and 
 {higher_or_lower(portfolio_sharpe_ratio, btc_sharpe)} than that of {btc} ({btc_sharpe})"""