import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


def skewness(r):
    #Calculates the skewness of a return series or dataframe
    d_r = r - r.mean()
    sigma_r = r.std(ddof=0) #setting degrees of freedom to be zero
    exp = (d_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    #calculates the kurtosis of a return series/ dataframe
    r_1 = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = ((r_1)**4).mean()
    return exp/sigma_r**4

def compound(r):
    return np.expm1(np.log1p(r).sum())

def annualized_return(r, periods_per_year=252):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
    #calculates annual returns given a monthly return series
    # annualized_returns = []
    # for column in r.columns:
    #     returns = r[column]
    #     total_returns=1
    #     for cumm_return in returns:
    #         total_returns *= (1 + cumm_return)
    #     total_returns = (total_returns) ** (252/ len(returns)) - 1
    #     annualized_returns.append(total_returns)
    # return annualized_returns


def annualize_vol(r):
    #calculates annualized volatility given a monthly return series
    return r.std()*(252**0.5)


def sharpe_ratio(r, riskfree_rate=0.03):
    #calculates the sharpe ratio given a monthly return series and risk free rate as 3 per cent
    rf_per_period = (1+riskfree_rate)**(1/252)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_return(excess_ret)
    ann_vol = annualize_vol(r)
    return ann_ex_ret/ann_vol

def drawdown(return_series: pd.Series):
    #output: a dataframe having columns which give drawdown, the wealth index and the maximum return value so far for every month
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


def semideviation(r):
    #calculates semi-deviation(deviation for negative returns) for a monthly return series
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")    

def var_gaussian(r, level=5, modified=False):
    #calculates the Value at risk for a return series and 5% confidence
    #For normal returns, use modified = False
    #For returns which are not normal, use modified = True (Cornish Fisher VaR is used)
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def portfolio_return(weights, returns):
    #calculates portfolio returns given returns of the assets and their weights
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    #calculates portfolio volatility given covariance of the assets and their weights
    return (weights.T @ covmat @ weights)**0.5

def minimize_vol(target_return, er, cov):
    #outputs weights for assets to minimize volatility for a particular return rate of the portfolio
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 

    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def msr(riskfree_rate, er, cov):
    #outputs weights for a portfolio having maximum sharpe ratio out of all possible portfolios
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    #gives optimal weights for a given return rate
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    #Plots the efficient frontier for a given portfolio of assets
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
    return ax



def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    
    #Runs a backtest of the CPPI strategy, given a set of returns for the risky asset
    # Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History

    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 

    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result

def max_drawdown(r):
    """Calculates the maximum drawdown of a return series."""
    wealth_index = (1 + r).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

def sortino_ratio(r, riskfree_rate=0.03):
    """Calculates the Sortino Ratio for a return series."""
    rf_per_period = (1 + riskfree_rate) ** (1 / 252) - 1
    excess_ret = r - rf_per_period
    downside_vol = semideviation(excess_ret)
    ann_ex_ret = annualized_return(excess_ret)
    return ann_ex_ret / downside_vol

# Update summary_stats to include them
def summary_stats(r, riskfree_rate=0.03):
    ann_r = annualized_return(r)
    ann_vol = annualize_vol(r)
    ex_ret = [i - riskfree_rate for i in ann_r]
    ann_sr = ex_ret / ann_vol
    skew = skewness(r)
    kurt = kurtosis(r)
    cf_var5 = var_gaussian(r, modified=True)
    mdd = max_drawdown(r)
    sortino = sortino_ratio(r, riskfree_rate)

    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Sharpe Ratio": ann_sr,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
    }).reset_index()






    

