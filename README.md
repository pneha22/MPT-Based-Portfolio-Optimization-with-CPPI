# Portfolio Optimization with MPT & CPPI

Hey! 👋  
This repo is a collection of my experiments with portfolio optimization. It’s mainly Python code from my notebook where I played with Modern Portfolio Theory (MPT) and the CPPI strategy to manage investments and risk.

## What’s Inside?

- **Modern Portfolio Theory (MPT):**  
  I used historical price/returns data to find the “best” mix of assets on the efficient frontier. Basically, I tried to maximize returns for a given risk, or minimize risk for a given return.  
  The logic: MPT uses covariance between assets to show how combining them can lower the overall risk, thanks to diversification.

- **Constant Proportion Portfolio Insurance (CPPI):**  
  I also tested the CPPI method, which is a way to protect your money from dropping below a certain floor, while still allowing for growth.  
  The gist: You set a “floor” value, and the algorithm calculates how much to keep in risky assets vs. safe ones (like cash or bonds), adjusting dynamically as the portfolio value changes.

## How It Works

- Load asset price data ( Yahoo Finance).
- Calculate daily returns and stats (mean, covariance, etc).
- Build the efficient frontier (MPT) and pick optimal weights.
- Run CPPI simulation where a floor value (minimum acceptable portfolio value) is defined and a multiplier determines how aggressively to allocate between risky and safe assets.
- Both MPT and CPPI are backtested and evaluated on returns, volatility, VaR, and drawdown, with plots of risk–return tradeoffs and wealth over time.

## Why?

I wanted to see how theory matches practical outcomes, and how much protection CPPI actually gives compared to traditional allocation.  


