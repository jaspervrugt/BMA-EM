# ####################################################################### #
#                                                                         #
# BBBBBBBBB    MMM        MMM      AAA          EEEEEEEEE  MMM        MMM #
# BBBBBBBBBB   MMMM      MMMM     AAAAA         EEEEEEEEE  MMMM      MMMM #
# BBB     BBB  MMMMM    MMMMM    AAA AAA        EEE        MMMMM    MMMMM #
# BBB     BBB  MMMMMM  MMMMMM   AAA   AAA       EEE        MMMMMM  MMMMMM #
# BBB    BBB   MMM MMMMMM MMM  AAA     AAA ---- EEEEE      MMM MMMMMM MMM #
# BBB    BBB   MMM  MMMM  MMM  AAAAAAAAAAA ---- EEEEE      MMM  MMMM  MMM #
# BBB     BBB  MMM   MM   MMM  AAA     AAA      EEE        MMM   MM   MMM #
# BBB     BBB  MMM        MMM  AAA     AAA      EEE        MMM        MMM #
# BBBBBBBBBB   MMM        MMM  AAA     AAA      EEEEEEEEE  MMM        MMM #
# BBBBBBBBB    MMM        MMM  AAA     AAA      EEEEEEEEE  MMM        MMM #
#                                                                         #
# ####################################################################### #
#                                                                         #
#  Ensemble Bayesian model averaging provides a methodology to explicitly #
#  handle conceptual model uncertainty in the interpretation and analysis #
#  of environmental systems. This method combines the predictive          #
#  capabilities of multiple different models and jointly assesses their   #
#  uncertainty. The probability density function (pdf) of the quantity of #
#  interest predicted by Bayesian model averaging is essentially a        #
#  weighted average of individual pdf's predicted by a set of different   #
#  models that are centered around their forecasts. The weights assigned  #
#  to each of the models reflect their contribution to the forecast skill #
#  over the training period                                               #
#                                                                         #
#  This code uses the Expectation Maximization algorithm for BMA model    #
#  training. This version for BMA model training assumes the predictive   #
#  PDF's of the ensemble members to equal a normal, lognormal, truncated  #
#  normal or gamma distribution with constant or nonconstant variance.    #
#  Please refer to the MODELAVG Package for a larger suite of predictive  #
#  PDF's of the ensemble members and other added functionalities & visual #
#  output to the screen                                                   #
#                                                                         #
#  SYNOPSIS: [beta,sigma,loglik,it] = BMA_EM(D,y,options)                 #
#            [beta,sigma,loglik,it] = BMA_EM(D,y,options,errtol)          #
#            [beta,sigma,loglik,it] = BMA_EM(D,y,options,errtol,maxit)    #
#   where                                                                 #
#    D         [input] nxK matrix forecasts ensemble members              #
#    y         [input] nx1 vector with verifying observations             #
#    options   [input] structure BMA algorithmic variables                #
#     .PDF             string: conditional PDF for BMA method (MANUAL)    #
#                       = 'normal'     = normal distribution              #
#                       = 'lognormal'  = lognormal distribution           #
#                       = 'tnormal'    = truncated normal ( > 0)          #
#                       = 'gamma'      = gamma distribution               #
#     .VAR             string: variance treatment BMA method (MANUAL)     #
#                       =  '1'          = constant group variance         #
#                       =  '2'          = constant individual variance    #
#                       =  '3'          = nonconstant group variance      #
#                       =  '4'          = nonconstant ind. variance       #
#     .alpha           significance level for prediction limits BMA model #
#                       = [0.5 0.7 0.9 0.95 0.99]                         #
#   errtol    [input] OPTIONAL: error tolerance: default = 1e-4           #
#   maxit     [input] OPTIONAL: maximum # iterations: default = 10000     #
#   beta      [outpt] (1 x K)-vector of BMA weights                       #
#   sigma     [outpt] (1 x K)-vector of BMA standard deviations - or      #
#   c         [outpt] (1 x K)-vector of multipliers c: sigma = c*|D|      #
#   loglik    [outpt] BMA log-likelihood                                  #
#   it        [outpt] number of iterations to reach convergence           #
#                                                                         #
# ####################################################################### #
#                                                                         #
#  LITERATURE                                                             #
#   Vrugt, J.A. (2024), Distribution-Based Model Evaluation and           #
#       Diagnostics: Elicitability, Propriety, and Scoring Rules for      #
#       Hydrograph Functionals, Water Resources Research, 60,             #
#       e2023WR036710,                                                    #
#           https://doi.org/10.1029/2023WR036710                          #
#   Vrugt, J.A. (2015), Markov chain Monte Carlo simulation using the     #
#       DREAM software package: Theory, concepts, and MATLAB              #
#       implementation, Environmental Modeling and Software, 75,          #
#       pp. 273-316                                                       #
#   Diks, C.G.H., and J.A. Vrugt (2010), Comparison of point forecast     #
#       accuracy of model averaging methods in hydrologic applications,   #
#       Stochastic Environmental Research and Risk Assessment, 24(6),     #
#       809-820, https://doi.org/10.1007/s00477-010-0378-z                #
#   Vrugt, J.A., C.G.H. Diks, and M.P. Clark (2008), Ensemble Bayesian    #
#       model averaging using Markov chain Monte Carlo sampling,          #
#       Environmental Fluid Mechanics, 8(5-6), 579-595,                   #
#           https://doi.org/10.1007/s10652-008-9106-3                     #
#   Wöhling, T., and J.A. Vrugt (2008), Combining multi-objective         #
#       optimization and Bayesian model averaging to calibrate forecast   #
#       ensembles of soil hydraulic models, Water Resources Research,     #
#       44, W12432,                                                       #
#          https://doi.org/10.1029/2008WR007154                           #
#   Vrugt, J.A., and B.A. Robinson (2007), Treatment of uncertainty using #
#       ensemble methods: Comparison of sequential data assimilation and  #
#       Bayesian model averaging, Water Resources Research, 43, W01411,   #
#           https://doi.org/10.1029/2005WR004838                          #
#   Vrugt, J.A., M.P. Clark, C.G.H. Diks, Q. Duan, and B.A.               #
#       Robinson (2006), Multi-objective calibration of forecast          #
#       ensembles using Bayesian Model Averaging, Geophysical Research    #
#       Letters, 33, L19817,                                              #
#           https://doi.org/10.1029/2006GL027126                          #
#   Raftery, A.E., T. Gneiting, F. Balabdaoui, and M. Polakowski (2005),  #
#       Using Bayesian model averaging to calibrate forecast ensembles,   #
#       Monthly Weather Revivew, 133, 1155–1174                           #
#   Raftery, A.E., and Y. Zheng (2003), Long-run performance of Bayesian  #
#       model averaging, Journal of the American Statistical Association, #
#       98, 931–938                                                       #
#   Raftery, A.E., D. Madigan, and J.A. Hoeting (1997), Bayesian model    #
#       averaging for linear regression models, Journal of the American   #
#       Statistical Association, 92, 179–191                              #
#                                                                         #
# ######################################################################  #
#                                                                         #
#  BUILT-IN CASE STUDIES                                                  #
#    Example 1   24-hour forecasts of river discharge                     #
#    Example 2   48-forecasts of sea surface temperature                  #
#    Example 3   48-forecasts of sea surface pressure                     #
#    Example 11  Forecasts of water levels                                #
#    Example 12  Hydrologic modeling                                      #
#    Example 13  Flood modeling                                           #
#                                                                         #
# ####################################################################### #
#                                                                         #
# COPYRIGHT (c) 2024  the author                                          #
#                                                                         #
#   This program is free software: you can modify it under the terms      #
#   of the GNU General Public License as published by the Free Software   #
#   Foundation, either version 3 of the License, or (at your option)      #
#   any later version                                                     #
#                                                                         #
#   This program is distributed in the hope that it will be useful, but   #
#   WITHOUT ANY WARRANTY; without even the implied warranty of            #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      #
#   General Public License for more details                               #
#                                                                         #
# ####################################################################### #
#                                                                         #
#  PYTHON CODE                                                            #
#  © Written by Jasper A. Vrugt                                           #
#    Center for NonLinear Studies                                         #
#    Los Alamos National Laboratory                                       #
#    University of California Irvine                                      #
#  Version 0.5    July 2006                                               #
#  Version 1.0    January 2008  -- code cleaned and added examples        #
#  Version 1.1    July 2014     -- added quantiles                        #
#  Version 2.0    July 2024     -- added many more functionalities        #
#                                                                         #
# ####################################################################### #

import numpy as np
import scipy.special    # for the gamma function - not gamma distribution!
import scipy.special as sp
from scipy.special import erf, lambertw, gammainc
from scipy.stats import norm, lognorm, truncnorm, gamma, weibull_min, genextreme, genpareto
from scipy.optimize import fsolve
from scipy import stats

def BMA_loglik(x, beta, Y, D, n, K, options, flag):
    """
    Computes the log-likelihood of the BMA model.
    """
    if np.any(x < 0):
        return np.inf #, None  # Weights and variances/multipliers must not be negative
    
    if flag == 1:  # fminsearch implementation
        if options['VAR'] == '1':  # constant: sigma = x (scalar)
            sigma = x * np.ones((n, K))
        elif options['VAR'] == '2':  # constant: sigma = x (vector)
            sigma = np.tile(x, (n, 1))
        elif options['VAR'] == '3':  # nonconstant: c = x (scalar)
            sigma = x * np.abs(D)
        elif options['VAR'] == '4':  # nonconstant: c = x (vector)
            sigma = np.tile(x, (n, 1)) * np.abs(D)
    else:  # sigma is already known
        if options['VAR'] == '1':  # constant: sigma = x (scalar)
            sigma = x * np.ones((n, K))
        elif options['VAR'] == '2':  # constant: sigma = x (vector)
            sigma = np.tile(x, (n, 1))
        else:
            sigma = x

    # Compute likelihood
    if options['PDF'] == 'normal':
        L = np.exp(-0.5 * ((Y - D) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    elif options['PDF'] == 'lognormal':
        A = sigma ** 2 / (D ** 2)
        S2 = np.log(A + 1)
        S = np.sqrt(S2)
        mu = np.log(np.abs(D)) - S2 / 2
        L = np.exp(-0.5 * ((np.log(Y) - mu) / S) ** 2) / (np.sqrt(2 * np.pi) * Y * S)
    elif options['PDF'] == 'gamma':
        mu = np.abs(D)
        A = mu ** 2 / sigma ** 2
        B = sigma ** 2 / mu
        Z = Y / B
        U = (A - 1) * np.log(Z) - Z - sp.gammaln(A)
        L = np.exp(U) / B
    elif options['PDF'] == 'tnormal':
        mu = np.maximum(D, np.finfo(float).eps)
        a = 0
        b = np.inf
        L = tnormpdf(Y, mu, sigma, a, b)

    # Return BMA log-likelihood
    ell = -np.sum(np.log(np.dot(L, beta.T)))
    if flag == 1:
        return ell
    else:
        return ell,L


def fast_CRPS(beta, sigma, D, y):
    ## ##################################################################### ##
    ## Computes the continues ranked probability score: Fast and approximate ##
    ## for a normal predictive PDF with constant group/individual variances  ##
    ## Note: the CRPS is treated as negative oriented: smaller is better     ##
    ## This function is OBSOLETE, use BMA_quantile instead. This uses an     ##
    ## exact quantile solution to the CRPS using built-in integral function  ##
    ##                                                                       ##
    ## SYNOPSIS: crps = fast_CRPS(beta,sigma,D,y)                            ##
    ##  where                                                                ##
    ##   beta      [input] (1 x K)-vector of BMA weights                     ##
    ##   sigma     [input] (1 x K)-vector of BMA standard deviations - or    ##
    ##   D         [input]  REQUIRED: n x K matrix of ensemble forecasts     ##
    ##   y         [input]  REQUIRED: n x 1 vector of verifying observations ##
    ##   crps      [outpt] Continuous Ranked Probability Score for BMA model ##
    ##                                                                       ##
    ## Note: the CRPS is positively-oriented!! (larger is better)            ##
    ##                                                                       ##
    ## (c) Written by Jasper A. Vrugt, Feb 2006                              ##
    ## Los Alamos National Laboratory                                        ##
    ##                                                                       ##
    ## ##################################################################### ##
    
    n, K = D.shape  # Number of observations and models
    # Ensure sigma is a NumPy array, even if it's passed as a list or scalar
    sigma = np.asarray(sigma)

    # Set up the variance
    if len(sigma) == 1:
        s2 = sigma**2 * np.ones(K)
    else:
        s2 = sigma**2

    s = np.sqrt(s2)     # Standard deviations
    crps = np.zeros(n)  # Array to store CRPS values
    # Loop over each observation
    for t in range(n):
        first_sum = 0
        second_sum = 0
        
        # Compute the first sum (double sum of CRPS)
        for k in range(K):
            for j in range(K):
                ts2 = s2[k] + s2[j]  # Total variance
                ts = np.sqrt(ts2)    # Total standard deviation
                dD = D[t, k] - D[t, j]  # Difference in forecasts
                first_sum += beta[k] * beta[j] * absexp(dD, ts)
        
        # Compute the second sum (single sum of CRPS)
        for k in range(K):
            ts2 = s2[k]
            ts = np.sqrt(ts2)
            dy = D[t, k] - y[t]  # Difference from the observation
            second_sum += beta[k] * absexp(dy, ts)
        
        # Compute CRPS for this observation
        crps[t] = -0.5 * first_sum + second_sum
    
    # Return the negative mean absolute CRPS (then CRPS is positively-oriented)
    return -np.abs(np.mean(crps))


def absexp(mu, sigma):
    """
    Computes the absolute exponential function for CRPS.
    
    Parameters:
    - mu: mean of the normal distribution
    - sigma: standard deviation of the normal distribution
    
    Returns:
    - y: scalar result
    """
    
    return (1 / np.sqrt(np.pi)) * sigma * np.sqrt(2) * np.exp(-0.5 * (mu**2) / (sigma**2)) \
        + mu * erf(0.5 * mu * np.sqrt(2) / sigma)


def BMA_quantile(x, D, y, options):
    # ####################################################################### #
    # This function calculates the quantiles of BMA mixture distribution      #
    #                                                                         #
    # SYNOPSIS: [pred,pdf_y,cdf_y] = BMA_quantile(x,D,y,options)              #
    #  where                                                                  #
    #   x         [input] 1xd vector with maximum likelihood BMA parameters   #
    #   D         [input] nxK matrix with forecasts of ensemble members       #
    #   y         [input] nx1 vector with verifying observations              #
    #   options   [input] structure with BMA algorithmic variables            #
    #   plimit    [outpt] lower and upper end of alpha prediction interval    #
    #   pdf_y     [outpt] nx1 vector with pdf of BMA distribution at y        #
    #   cdf_y     [outpt] nx1 vector with cdf of BMA distribution at y        #
    #                                                                         #
    # (c) Written by Jasper A. Vrugt, Feb 2006                                #
    # University of California Irvine                                         #
    #                                                                         #
    # ####################################################################### #

    n, K = D.shape                  # Unpack number of observations
    beta = x[:K]                    # Unpack weights
    count = 0                       # Set counter to zero
    pdf_y = np.nan * np.ones(n)     # Initialize PDF observations
    cdf_y = np.nan * np.ones(n)     # Initialize CDF observations

    # Unpack alpha values
    g = 1 - np.array(options['alpha'])
    gam = np.sort([(1 - g) / 2, (1 - g) / 2 + g])  # Quantile levels
    # added by JAV to make sure that gam is a row vector with nA elements
    gam = gam.reshape(-1)
    nA = len(gam)

    # Set variance based on the options.VAR setting
    if options['VAR'] == '1':       # Common constant variance
        S = x[K] * np.ones((n, K))
    elif options['VAR'] == '2':     # Individual constant variance
        S = np.tile(x[K:2*K], (n, 1))
    elif options['VAR'] == '3':     # Common non-constant variance
        c = x[K]
        S = c * np.abs(D)
    elif options['VAR'] == '4':     # Individual non-constant variance
        c = x[K:2*K]
        S = np.multiply(c[:, None], np.abs(D))
    else:
        raise ValueError("Unknown variance option")

    # Ensure S is positive
    S = np.abs(S)

    # Conditional distribution case
    if options['PDF'] == 'normal':
        Mu = D  # Mean of normal distribution

        def PDF(x, i):
            result = beta[0] * norm.pdf(x, Mu[i, 0], S[i, 0])
            for k in range(1, K):
                result += beta[k] * norm.pdf(x, Mu[i, k], S[i, k])
            return result

        def CDF(x, i):
            result = beta[0] * norm.cdf(x, Mu[i, 0], S[i, 0])
            for k in range(1, K):
                result += beta[k] * norm.cdf(x, Mu[i, k], S[i, k])
            return result

    elif options['PDF'] == 'lognormal':
        logn_impl = 2
        if logn_impl == 1:
            Mu = np.log(np.abs(D)) - S**2 / 2  # Mean for lognormal distribution
        elif logn_impl == 2:
            sigma2_fxs = S**2
            A = sigma2_fxs / (D**2)
            S2 = np.log(A + 1)
            S = np.sqrt(S2)
            Mu = np.log(np.abs(D)) - S2 / 2
        
        def PDF(x, i):
            result = beta[0] * lognorm.pdf(x, S[i, 0], scale=np.exp(Mu[i, 0]))
            for k in range(1, K):
                result += beta[k] * lognorm.pdf(x, S[i, k], scale=np.exp(Mu[i, k]))
            return result

        def CDF(x, i):
            result = beta[0] * lognorm.cdf(x, S[i, 0], scale=np.exp(Mu[i, 0]))
            for k in range(1, K):
                result += beta[k] * lognorm.cdf(x, S[i, k], scale=np.exp(Mu[i, k]))
            return result

    elif options['PDF'] == 'tnormal':
        Mu = np.abs(D)  # Mode truncated normal
        a = 0           # Lower end point
        b = np.inf      # Upper end point
        Alfa = (a - Mu) / S
        Beta = (b - Mu) / S

        def PDF(x, i):
            result = beta[0] * truncnorm.pdf(x, Alfa[i,0], Beta[i,0], loc = Mu[i, 0], scale = S[i, 0])
            for k in range(1, K):
                result += beta[k] * truncnorm.pdf(x, Alfa[i,k], Beta[i,k], loc = Mu[i, k], scale = S[i, k])
            return result

        def CDF(x, i):
            result = beta[0] * truncnorm.cdf(x, Alfa[i,0], Beta[i,0], loc = Mu[i, 0], scale = S[i, 0])
            for k in range(1, K):
                result += beta[k] * truncnorm.cdf(x, Alfa[i,k], Beta[i,k], loc = Mu[i, k], scale = S[i, k])
            return result

    elif options['PDF'] == 'gen_normal':
        Mu = D
        d = len(x)
        if options['TAU'] == '1':
            tau = x[d] * np.ones((n, K))
        elif options['TAU'] == '2':
            tau = np.tile(x[d-K:d], (n, 1))

        def PDF(x, i):
            result = beta[0] * gnormpdf(x, Mu[i, 0], S[i, 0], tau[i,0])
            for k in range(1, K):
                result += beta[k] * gnormpdf(x, Mu[i, k], S[i, k], tau[i,k])
            return result

        def CDF(x, i):
            result = beta[0] * gnormcdf(x, Mu[i, 0], S[i, 0], tau[i,0])
            for k in range(1, K):
                result += beta[k] * gnormcdf(x, Mu[i, k], S[i, k], tau[i,k])
            return result

    elif options['PDF'] == 'gamma':
        Mu = np.abs(D)      # Mean of gamma distribution
        A = (Mu**2) / S**2  # Shape parameter
        B = S**2 / Mu       # Scale parameter

        def PDF(x, i):
            result = beta[0] * gamma.pdf(x, A[i, 0], scale = B[i, 0])
            for k in range(1, K):
                result += beta[k] * gamma.pdf(x, A[i, k], scale = B[i, k])
            return result

        def CDF(x, i):
            result = beta[0] * gamma.cdf(x, A[i, 0], scale = B[i, 0])
            for k in range(1, K):
                result += beta[k] * gamma.cdf(x, A[i, k], scale = B[i, k])
            return result

    elif options['PDF'] == 'weibull':
        wbl_impl = 2
        if wbl_impl == 1:
            Kk = S
            Lambda = np.abs(D) / scipy.special.gamma(1 + 1 / Kk)
            Lambda = np.maximum(Lambda, np.finfo(float).eps)  # Avoid zero Lambda
        elif wbl_impl == 2:
            Lambda = S
            X = np.abs(D) / Lambda
            c = np.sqrt(2 * np.pi) / np.exp(1) - scipy.special.gamma(1.461632)
            Lx = lambda x: np.log((x + c) / np.sqrt(2 * np.pi))
            A = Lx(X)
            B = A / np.real(lambertw(A / np.exp(1))) + 1 / 2
            Kk = 1 / (B - 1)

        def PDF(x, i):
            result = beta[0] * weibull_min.pdf(x, Kk[i, 0], scale = Lambda[i, 0])
            for k in range(1, K):
                result += beta[k] * weibull_min.pdf(x, Kk[i, k], scale = Lambda[i, k])
            return result

        def CDF(x, i):
            result = beta[0] * weibull_min.cdf(x, Kk[i, 0], scale = Lambda[i, 0])
            for k in range(1, K):
                result += beta[k] * weibull_min.cdf(x, Kk[i, k], scale = Lambda[i, k])
            return result

    elif options['PDF'] == 'gev':
        d = len(x)
        if options['TAU'] == '1':
            xi = x[d] * np.ones((n, K))
        elif options['TAU'] == '2':
            xi = np.tile(x[d-K:d], (n, 1))

        g = lambda a, xi: scipy.special.gamma(1 - a * xi)
        Mu = D - (g(1, xi) - 1) * S / xi

        # Handle case where xi is very close to zero
        Eul_constant = 0.577215664901532860606512090082
        Mu[np.abs(xi) < np.finfo(float).eps] = D[np.abs(xi) < np.finfo(float).eps] - S[np.abs(xi) < np.finfo(float).eps] * Eul_constant

        def PDF(x, i):
            result = beta[0] * genextreme.pdf(x, xi[i, 0], loc = Mu[i, 0], scale = S[i, 0])
            for k in range(1, K):
                result += beta[k] * genextreme.pdf(x, xi[i, k], loc = Mu[i, k], scale = S[i, k])
            return result

        def CDF(x, i):
            result = beta[0] * genextreme.cdf(x, xi[i, 0], loc = Mu[i, 0], scale = S[i, 0])
            for k in range(1, K):
                result += beta[k] * genextreme.cdf(x, xi[i, k], loc = Mu[i, k], scale = S[i, k])
            return result

    elif options['PDF'] == 'gpareto':
        d = len(x)
        if options['TAU'] == '1':
            kk = x[d] * np.ones((n, K))
        elif options['TAU'] == '2':
            kk = np.tile(x[d-K:d], (n, 1))

        Theta = np.abs(D) - S / (1 - kk)

        def PDF(x, i):
            result = beta[0] * genpareto.pdf(x, kk[i,0], loc = Theta[i, 0], scale = S[i, 0])
            for k in range(1, K):
                result += beta[k] * genpareto.pdf(x, kk[i, k], loc = Theta[i, k], scale = S[i, k])
            return result
        
        def CDF(x, i):
            result = beta[0] * genpareto.cdf(x, kk[i,0], loc = Theta[i, 0], scale = S[i, 0])
            for k in range(1, K):
                result += beta[k] * genpareto.cdf(x, kk[i, k], loc = Theta[i, k], scale = S[i, k])
            return result

    # Compute PDF and CDF at measured values
    for i in range(n):
        pdf_y[i] = PDF(y[i], i)
        cdf_y[i] = CDF(y[i], i)

    # Now activate the mixture CDF with alpha for root finding
    def Fx(x, i, gam):
      #  if x < 0:
      #      return 1e10
      #  else:
        return CDF(x, i) - gam

    # Now loop over each observation and determine BMA quantiles
    plimit = np.nan * np.ones((n, nA))
    x = np.linspace(min(0, 2 * np.min(np.min(D), axis=0)), 2 * np.max(y), int(1e5))  # Define x values
    calc_method = 1  # Set calculation method (1 or 2)
    for i in range(n):
        if i % (n // 25) == 0:
            if i > 0:
                print(f'BMA quantile calculation, {100*(i/n):.2f}% done', end = '\r')

        if calc_method == 1:  # Exact answer using root finding of CDF, but slower
            for z in range(nA):
                # Find root using the fzero equivalent in Python (optimize.root_scalar)
                # plimit[i, z] = fsolve(lambda x: Fx(x, i, gam[z]), abs(np.mean(D[i, :])))
                plimit[i, z], info, ier, msg = fsolve(lambda x: Fx(x, i, gam[z]), y[i], full_output=True)
                if ier != 1:
                    plimit[i, z] = fsolve(lambda x: Fx(x, i, gam[z]), abs(np.mean(D[i, :])))
                # be careful => initial value must make sense, poor convergence shows in the quantile plot
                # MATLAB appears much more robust here, meaning the initial value of np.mean(abs(D[i, :])) works well across forecast PDFs and different variables/time series
                # Must check whether fsolve converged properly [= reported in output argument] and then use linear interpolation as alternative
                # plimit[i, z], info, ier, msg = fsolve(lambda x: Fx(x, i, gam[z]), abs(np.mean(D[i, :])), full_output=True)
                # check ier and do linear interpolation!
        elif calc_method == 2:  # Approximate answer using linear interpolation
            cdf_x = Fx(x, i, 0)  # Get the CDF values
            ii = np.diff(cdf_x) > 0  # Find where the CDF is increasing
            ii = np.concatenate(([True], ii))
            plimit[i, :nA] = np.interp(gam, cdf_x[ii], x[ii])  # Linear interpolation
    
    print("\n BMA quantile calculation complete.")

    return plimit, pdf_y, cdf_y


def BMA_crps(x, D, y, options):
    # ####################################################################### #
    # This function computes the Continuous Ranked Probability Score for      #
    # BMA mixture distribution using numerical (trapezoidal) integration of   #
    # the quantile formulation:                                               #
    #     CRPS(P,w) = w(1 - 2F_P(w)) + 2\int_{0}^{1}tau F_P^{-1}(tau)d tau    #
    #                 - 2\int_{F_P(w)}^{1} F_P^{-1}(tau)d tau                 #
    #                                                                         #
    # SYNOPSIS: [mean_crps,crps,num_nan] = BMA_crps(x,D,Y,options)            #
    #  where                                                                  #
    #   x         [input] 1xd vector with maximum likelihood BMA parameters   #
    #   D         [input] nxK matrix with forecasts of ensemble members       #
    #   y         [input] nx1 vector with verifying observations              #
    #   options   [input] structure with BMA algorithmic variables            #
    #   crps      [outpt] nx1 vector with CRPS values of BMA mixture CDF      #
    #                                                                         #
    # Reference:                                                              #
    #                                                                         #
    # Note: the CRPS is positively-oriented!! (larger is better)              #
    #                                                                         #
    # (c) Written by Jasper A. Vrugt, Feb 2022                                #
    # University of California Irvine                                         #
    #                                                                         #
    # ####################################################################### #

    n, K = D.shape              # Unpack the number of observations
    beta = x[:K]                # Unpack weights
    count = 0                   # Set counter to zero
    crps = np.full(n, np.nan)   # Initialize CRPS of BMA mixture CDF

    # Define tau points
    P1 = np.concatenate([np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05]), np.arange(0.05, 0.45, 0.05)])
    P = np.concatenate([P1, [0.5], 1 - P1[::-1]])  # tau points for integration
    nP = len(P)

    # Step 1: Determine the variance based on options.VAR
    if options['VAR'] == '1':       ## common constant variance
        S = x[K] * np.ones((n, K))
    elif options['VAR'] == '2':     ## individual constant variance
        S = np.tile(x[K:2*K], (n, 1))
    elif options['VAR'] == '3':     ## common non-constant variance
        c = x[K]
        S = c * np.abs(D)
    elif options['VAR'] == '4':     ## individual non-constant variance
        c = x[K:2*K]
        S = c[:, np.newaxis] * np.abs(D)
    else:
        raise ValueError("Unknown variance option")

    S = np.maximum(S, np.finfo(float).eps)  # Ensure S >= eps
    S2 = S ** 2  # variance matrix

    # Step 2: Define the CDF of the mixture distribution
    if options['PDF'] == 'normal':          # normal PDF
        Mu = D
        # Create CDF function for normal mixture
        def F_P(x, i, p):
            F = beta[0] * norm.cdf(x, Mu[i, 0], S[i, 0])
            for k in range(1, K):
                F += beta[k] * norm.cdf(x, Mu[i, k], S[i, k])
            return F - p

    elif options['PDF'] == 'lognormal':     # lognormal PDF
        Mu = np.log(np.abs(D)) - S2 / 2
        # Create CDF function for lognormal mixture
        def F_P(x, i, p):
            F = beta[0] * lognorm.cdf(x, S[i, 0], scale = np.exp(Mu[i, 0]))
            for k in range(1, K):
                F += beta[k] * lognorm.cdf(x, S[i, k], scale = np.exp(Mu[i, k]))
            return F - p

    elif options['PDF'] == 'tnormal':       # truncated normal PDF
        Mu = np.abs(D)
        a, b = 0, np.inf  # Lower and upper bounds for truncation
        Alfa = (a - Mu) / S
        Beta = (b - Mu) / S
        # Create CDF function for truncated normal mixture
        def F_P(x, i, p):
            # F = beta[0] * tnormcdf(x, Mu[i, 0], S[i, 0], a, b)
            F = beta[0] * truncnorm.cdf(x, Alfa[i,0], Beta[i,0], loc = Mu[i, 0], scale = S[i, 0])
            for k in range(1, K):
                # F += beta[k] * tnormcdf(x, Mu[i, k], S[i, k], a, b)
                F += beta[k] * truncnorm.cdf(x, Alfa[i,k], Beta[i,k], loc = Mu[i, k], scale = S[i, k])
            return F - p

    elif options['PDF'] == 'gen_normal':       # generalized normal PDF
        d = len(x)
        if options['TAU'] == '1':
            tau = x[d] * np.ones((n, K))
        elif options['TAU'] == '2':
            tau = np.tile(x[d-K:d], (n, 1))
        Mu = D
        # Create CDF function for generalized normal mixture
        def F_P(x, i, p):
            F = beta[0] * gnormcdf(x, Mu[i, 0], S[i, 0], tau[i,0])
            for k in range(1, K):
                F += beta[k] * gnormcdf(x, Mu[i, k], S[i, k], tau[i,k])
            return F - p

    elif options['PDF'] == 'gamma':         # gamma PDF
        Mu = np.abs(D)
        A = Mu**2 / S2
        B = S2 / Mu
        # Create CDF function for gamma mixture
        def F_P(x, i, p):
            F = beta[0] * gamma.cdf(x, A[i, 0], scale = B[i, 0])
            for k in range(1, K):
                F += beta[k] * gamma.cdf(x, A[i, k], scale = B[i, k])
            return F - p

    elif options['PDF'] == 'weibull':       # Weibull distribution parameters
        wbl_impl = 2
        if wbl_impl == 1:
            Kk = S
            Lambda = np.abs(D) / scipy.special.gamma(1 + 1 / Kk)
            Lambda = np.maximum(Lambda, np.finfo(float).eps)
        elif wbl_impl == 2:
            Lambda = S
            X = np.abs(D) / Lambda
            c = np.sqrt(2 * np.pi) / np.exp(1) - scipy.special.gamma(1.461632)
            A = np.log((X + c) / np.sqrt(2 * np.pi))
            B = A / np.real(np.log(A / np.exp(1))) + 1 / 2
            Kk = 1 / (B - 1)

        # Create CDF function for Weibull mixture
        def F_P(x, i, p):
            F = beta[0] * weibull_min.cdf(x, Kk[i, 0], scale = Lambda[i, 0])
            for k in range(1, K):
                F += beta[k] * weibull_min.cdf(x, Kk[i, k], scale = Lambda[i, k])
            return F - p

    elif options['PDF'] == 'gev':           # Generalized Extreme Value (GEV) distribution
        d = len(x)
        if options['TAU'] == '1':  # Common tau
            xi = x[d] * np.ones((n, K))
        elif options['TAU'] == '2':  # Individual tau
            xi = np.tile(x[d-K:d], (n, 1))
        
        g = lambda a, xi: scipy.special.gamma(1 - a * xi)
        Mu = D - (g(1, xi) - 1) * S / xi
        
        # Handle case where xi is very close to zero
        Eul_constant = 0.577215664901532860606512090082
        Mu[np.abs(xi) < np.finfo(float).eps] = D[np.abs(xi) < np.finfo(float).eps] - S[np.abs(xi) < np.finfo(float).eps] * Eul_constant

        # Create CDF function for GEV mixture
        def F_P(x, i, p):
            F = beta[0] * genextreme.cdf(x, xi[i, 0], loc = Mu[i, 0], scale = S[i, 0])
            for k in range(1, K):
                F += beta[k] * genextreme.cdf(x, xi[i, k], loc = Mu[i, k], scale = S[i, k])
            return F - p        

    elif options['PDF'] == 'gpareto':       # generalized Pareto PDF
        d = len(x)
        if options['TAU'] == '1':           # Common tau
            kk = x[d] * np.ones((n, K))
        elif options['TAU'] == '2':         # Individual tau
            kk = np.tile(x[d-K:d], (n, 1))

        Theta = np.abs(D) - S / (1 - kk)

        # Create CDF function for generalized Pareto mixture
        def F_P(x, i, p):
            F = beta[0] * genpareto.cdf(x, kk[i,0], loc = Theta[i, 0], scale = S[i, 0])
            for k in range(1, K):
                F += beta[k] * genpareto.cdf(x, kk[i, k], loc = Theta[i, k], scale = S[i, k])
            return F - p

    # Step 3: Compute CRPS for each observation
    FinvP = np.nan * np.ones(nP)
    for i in range(n):
        if i % (n // 25) == 0:
            if i > 0:
                print(f'BMA CRPS calculation, {100*(i/n):.2f}% done', end='\r')

        # Solve for values of y so that F_P^-1(y) = P using fsolve
        for z in range(nP):
            if z == 0:
                # FinvP[z] = fsolve(lambda y: F_P(y, i, P[z]), abs(np.mean(D[i, :]))) # initial value OK in MATLAB
                FinvP[z], info, ier, msg = fsolve(lambda y: F_P(y, i, P[z]), abs(np.mean(D[i, :])), full_output=True) # initial value OK in MATLAB 
            else:
                # FinvP[z] = fsolve(lambda y: F_P(y, i, P[z]), FinvP[z-1])
                FinvP[z], info, ier, msg = fsolve(lambda y: F_P(y, i, P[z]), FinvP[z-1], full_output=True) 
            if ier != 1:  # back-up if not converged properly
                FinvP[z] = fsolve(lambda y: F_P(y, i, P[z]), y[i])

        # Evaluate mixture CDF at omega
        Fy = F_P(y[i], i, 0)  # F_P evaluated at observation y(i)
        ii = P > Fy
        if np.sum(ii) == 0:
            crps[i] = y[i] * (1 - 2 * Fy) + 2 * np.trapezoid(P * FinvP,P)
        else:
            # crps[i] = y[i] * (1 - 2 * Fy) + 2 * np.trapezoid(P * FinvP,P) - 2 * np.trapezoid(np.concatenate(y[i],FinvP[ii]),np.concatenate(Fy,P[ii]))
            crps[i] = y[i] * (1 - 2 * Fy) + 2 * np.trapezoid(P * FinvP,P) - \
                2 * np.trapezoid(np.concatenate([y[i].flatten(), FinvP[ii].flatten()]),np.concatenate([Fy.flatten(), P[ii].flatten()]))

    print("\n BMA_crps: CRPS calculation complete.")

    return crps


def Bias_correction(D, y, intcpt = 1):
    # ####################################################################### #
    # This function provides a linear correction of the ensemble members      #
    #                                                                         #
    # SYNOPSIS: [D_bc,a,b] = Bias_correction(D,y)                             #
    #           [D_bc,a,b] = Bias_correction(D,y,intcpt)                      #
    #  where                                                                  #
    #   D         [input] nxK matrix with forecasts of ensemble members       #
    #   y         [input] nx1 vector with verifying observations              #
    #   intcpt    [input] OPT: intercept (default: 1) or without (0)          #
    #   D_bc      [outpt] nxK matrix with bias-corrected forecasts            #
    #   a         [outpt] 1xK vector with intercept bias-corrected forecasts  #
    #   b         [outpt] 1xK vector with slope bias-corrected forecasts      #
    #                                                                         #
    # (c) Written by Jasper A. Vrugt, Feb 2022                                #
    # University of California Irvine 			        	                  #
    #                                                                         #
    # ####################################################################### #
    
    # Get the shape of D
    n, K = D.shape
    D_bc = np.full((n, K), np.nan)  # Initialize with NaNs
    
    # Initialize intercepts and slopes of linear bias correction functions
    a = np.zeros(K)
    b = np.zeros(K)
    
    # Perform linear regression for each ensemble member
    for k in range(K):
        if intcpt == 1:
            # Linear regression with intercept
            X = np.column_stack((np.ones(n), D[:, k]))  # Add intercept column
            ab = np.linalg.lstsq(X, y, rcond=None)[0]   # Solve for intercept and slope
            a[k], b[k] = ab
        else:
            # Linear regression without intercept
            b[k] = np.linalg.lstsq(D[:, k].reshape(-1, 1), y, rcond=None)[0][0]
        
        # Bias-corrected ensemble forecasts
        D_bc[:, k] = a[k] + b[k] * D[:, k]
    
    return D_bc, a, b


def gnormcdf(x, mu, alfa, beta):
    # Generalized normal CDF
    P = 0.5 + np.sign(x - mu) * (1 / (2 * scipy.special.gamma(1 / beta))) * gammainc(1 / beta, np.abs((x - mu) / alfa) ** beta) * scipy.special.gamma(1 / beta)

    return P


def gnormpdf(x, mu, alfa, beta):
    """
    Generalized normal probability density function (pdf).
    
    Parameters:
    - x : array_like
        Values at which the pdf should be evaluated.
    - mu : float
        Mean of the distribution.
    - alfa : float
        Scale parameter (standard deviation).
    - beta : float
        Shape (kurtosis) parameter.

    Returns:
    - Y : array_like
        The values of the generalized normal pdf evaluated at x.
    """
    # Calculate the generalized normal pdf
    Y = beta / (2 * alfa * scipy.special.gamma(1 / beta)) * np.exp(- (np.abs(x - mu) / alfa) ** beta)
    
    return Y


def gnormrnd(mu, alfa, beta, *args):
    """
    Generate random variables from the Generalized Normal Distribution.

    Parameters:
    - mu : Mean of the distribution
    - alfa : Standard deviation (scale parameter)
    - beta : Shape parameter that controls the kurtosis
    - *args : Additional arguments to specify the shape of the output array (e.g., M, N, ...)
    
    Returns:
    - R : Randomly generated numbers from the generalized normal distribution
    """
    # Check if the number of arguments is valid
    if len(args) < 1:
        raise ValueError('Too few inputs, dimensions of output must be provided')
    
    # Determine the shape of the output array
    sizeOut = args

    # Generate random uniform variables p
    p = np.random.rand(*sizeOut)

    # Generate random samples from the generalized normal distribution
    R = mu + np.sign(p - 0.5) * (alfa ** beta * gamma.ppf(2 * np.abs(p - 0.5), 1 / beta)) ** (1 / beta)
    # gamma.ppf works fine - at least when separately evaluated

    return R


def tnormpdf(x, mu, sigma, a, b):
    """
    Truncated normal PDF.
    """
    norm_pdf = norm.pdf(x, loc=mu, scale=sigma)
    norm_cdf_lower = norm.cdf(a, loc=mu, scale=sigma)
    norm_cdf_upper = norm.cdf(b, loc=mu, scale=sigma)
    norm_cdf = norm_cdf_upper - norm_cdf_lower
    return norm_pdf / norm_cdf
