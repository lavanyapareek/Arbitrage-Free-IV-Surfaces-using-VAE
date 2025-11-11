"""
Heston Model Implementation using QuantLib

This module provides a robust Heston stochastic volatility model implementation
using QuantLib for accurate and efficient option pricing.
"""

import numpy as np
import QuantLib as ql
from scipy.optimize import brentq


class HestonModelQL:
    """
    Heston stochastic volatility model using QuantLib.
    
    Parameters:
    -----------
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance
    sigma_v : float
        Volatility of volatility
    rho : float
        Correlation between asset and volatility
    v0 : float
        Initial variance
    r : float
        Risk-free rate (default: 0.067 for NIFTY)
    q : float
        Dividend yield (default: 0.0 for NIFTY)
    """
    
    def __init__(self, kappa, theta, sigma_v, rho, v0, r=0.067, q=0.0):
        # Convert to Python floats to avoid QuantLib type issues
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.r = float(r)
        self.q = float(q)
        
        # QuantLib setup
        self.calendar = ql.NullCalendar()
        self.day_count = ql.Actual365Fixed()
        self.calculation_date = ql.Date(1, 1, 2020)
        ql.Settings.instance().evaluationDate = self.calculation_date
        
        # Initialize QuantLib objects
        self._setup_ql_objects()
    
    def _setup_ql_objects(self):
        """Setup QuantLib objects for pricing"""
        # Spot price (normalized to 1.0)
        self.spot = 1.0
        
        # Risk-free and dividend curves
        self.risk_free_curve = ql.FlatForward(
            self.calculation_date, 
            self.r, 
            self.day_count
        )
        self.dividend_curve = ql.FlatForward(
            self.calculation_date, 
            self.q, 
            self.day_count
        )
        
        # Heston process
        self.heston_process = ql.HestonProcess(
            ql.YieldTermStructureHandle(self.risk_free_curve),
            ql.YieldTermStructureHandle(self.dividend_curve),
            ql.QuoteHandle(ql.SimpleQuote(self.spot)),
            self.v0,
            self.kappa,
            self.theta,
            self.sigma_v,
            self.rho
        )
        
        # Heston model
        self.heston_model = ql.HestonModel(self.heston_process)
        
        # Pricing engine (analytic)
        self.engine = ql.AnalyticHestonEngine(self.heston_model)
    
    def update_parameters(self, kappa, theta, sigma_v, rho, v0, r=None, q=None):
        """Update model parameters"""
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        
        if r is not None:
            self.r = r
        if q is not None:
            self.q = q
        
        # Reinitialize QuantLib objects
        self._setup_ql_objects()
    
    def check_feller_condition(self):
        """Check Feller condition: 2*kappa*theta > sigma_v^2"""
        return 2 * self.kappa * self.theta > self.sigma_v**2
    
    def price_call(self, S, K, tau):
        """
        Price European call option using QuantLib.
        
        Parameters:
        -----------
        S : float
            Spot price
        K : float or array
            Strike price(s)
        tau : float
            Time to maturity (years)
        
        Returns:
        --------
        float or array
            Call option price(s)
        """
        is_scalar = np.isscalar(K)
        K_array = np.atleast_1d(K)
        
        # Maturity date
        maturity_date = self.calculation_date + ql.Period(int(tau * 365), ql.Days)
        
        prices = []
        for strike in K_array:
            # Create option
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(strike))
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(self.engine)
            
            # Price
            try:
                price = option.NPV()
                prices.append(price)
            except:
                prices.append(0.0)
        
        prices = np.array(prices)
        return prices[0] if is_scalar else prices
    
    def price_ratio(self, log_moneyness, tau):
        """
        Compute price ratio c/(S_0 * exp(-r*tau)) for given log-moneyness.
        
        Parameters:
        -----------
        log_moneyness : float or array
            Log-moneyness values (log(K/S))
        tau : float
            Time to maturity
        
        Returns:
        --------
        float or array
            Price ratio(s)
        """
        is_scalar = np.isscalar(log_moneyness)
        lm_array = np.atleast_1d(log_moneyness)
        
        S = self.spot
        K_array = S * np.exp(lm_array)
        
        call_prices = self.price_call(S, K_array, tau)
        
        # Price ratio: c / (S * exp(-r*tau))
        discount_factor = np.exp(-self.r * tau)
        price_ratios = call_prices / (S * discount_factor)
        
        return price_ratios[0] if is_scalar else price_ratios
    
    def implied_volatility(self, S, K, tau, market_price):
        """
        Compute implied volatility from market price.
        
        Parameters:
        -----------
        S : float
            Spot price
        K : float
            Strike price
        tau : float
            Time to maturity
        market_price : float
            Market call price
        
        Returns:
        --------
        float
            Implied volatility
        """
        maturity_date = self.calculation_date + ql.Period(int(tau * 365), ql.Days)
        
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)
        
        # Use Black-Scholes-Merton process for IV calculation
        flat_vol = ql.BlackConstantVol(
            self.calculation_date,
            self.calendar,
            0.2,  # Initial guess
            self.day_count
        )
        
        bsm_process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(self.dividend_curve),
            ql.YieldTermStructureHandle(self.risk_free_curve),
            ql.BlackVolTermStructureHandle(flat_vol)
        )
        
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        try:
            iv = option.impliedVolatility(
                market_price,
                bsm_process,
                1e-6,  # Accuracy
                100,   # Max iterations
                0.001, # Min vol
                2.0    # Max vol
            )
            return iv
        except:
            # Fallback to numerical method
            from scipy.stats import norm
            
            def bs_call_price(sigma):
                d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
                d2 = d1 - sigma * np.sqrt(tau)
                return S * np.exp(-self.q * tau) * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)
            
            def objective(sigma):
                return bs_call_price(sigma) - market_price
            
            try:
                return brentq(objective, 0.001, 5.0)
            except:
                return 0.2  # Default guess


class TimeVaryingHestonModelQL:
    """
    Time-varying Heston model using QuantLib.
    
    Parameters can vary across maturities to capture term structure effects.
    """
    
    def __init__(self, taus, params_list, r=0.067, q=0.0):
        """
        Parameters:
        -----------
        taus : array
            Array of maturities
        params_list : list of dict
            List of parameter dictionaries, one per maturity
            Each dict: {'kappa', 'theta', 'sigma_v', 'rho', 'v0'}
        r : float
            Risk-free rate
        q : float
            Dividend yield
        """
        self.taus = np.array(taus)
        self.params_list = params_list
        self.r = r
        self.q = q
        
        # Create Heston models for each maturity
        self.models = []
        for params in params_list:
            model = HestonModelQL(
                kappa=params['kappa'],
                theta=params['theta'],
                sigma_v=params['sigma_v'],
                rho=params['rho'],
                v0=params['v0'],
                r=r,
                q=q
            )
            self.models.append(model)
    
    def price_surface(self, log_moneyness_grid):
        """
        Generate price ratios for entire surface.
        
        Parameters:
        -----------
        log_moneyness_grid : array (n_maturities, n_strikes)
            Grid of log-moneyness values
        
        Returns:
        --------
        array (n_maturities, n_strikes)
            Price ratios
        """
        n_maturities = len(self.taus)
        n_strikes = log_moneyness_grid.shape[1]
        
        price_ratios = np.zeros((n_maturities, n_strikes))
        
        for i in range(n_maturities):
            price_ratios[i, :] = self.models[i].price_ratio(
                log_moneyness_grid[i, :], 
                self.taus[i]
            )
        
        return price_ratios
    
    def check_all_feller_conditions(self):
        """Check Feller condition for all maturities"""
        results = []
        for i, model in enumerate(self.models):
            results.append({
                'maturity': self.taus[i],
                'feller_satisfied': model.check_feller_condition(),
                'condition_value': 2 * model.kappa * model.theta - model.sigma_v**2
            })
        return results


# Backward compatibility: alias to QuantLib version
HestonModel = HestonModelQL
TimeVaryingHestonModel = TimeVaryingHestonModelQL
