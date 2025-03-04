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

# FURTHER CHECKING                                                        
#  Website:  http://faculty.sites.uci.edu/jasper                          
#  Papers: http://faculty.sites.uci.edu/jasper/publications/              
#  Google Scholar: https://scholar.google.com/citations?user=zkNXecUAAAAJ&hl=nl                                 

import numpy as np
import os, sys
from scipy.optimize import minimize

example_dir = os.getcwd()					                    # Add current directory to Python path
if example_dir not in sys.path:
    sys.path.append(example_dir)

parent_dir = os.path.abspath(os.path.join(example_dir, '..'))   # Go up one directory
sys.path.append(os.path.join(parent_dir, 'miscellaneous'))	    # Add miscellaneous directory to Python path
from BMA_EM_functions import *				                    # Import functions

# Main Program
def BMA_EM(D, y, options, errtol = 1e-4, maxit = 10000):
    
    # Print header information
    print('  -----------------------------------------------------------------------            ')
    print('  BBBBBBBBB    MMM        MMM      AAA          EEEEEEEEE  MMM        MMM            ')
    print('  BBBBBBBBBB   MMMM      MMMM     AAAAA         EEEEEEEEE  MMMM      MMMM            ')
    print('  BBB     BBB  MMMMM    MMMMM    AAA AAA        EEE        MMMMM    MMMMM            ')
    print('  BBB     BBB  MMMMMM  MMMMMM   AAA   AAA       EEE        MMMMMM  MMMMMM            ')
    print('  BBB    BBB   MMM MMMMMM MMM  AAA     AAA ---- EEEEE      MMM MMMMMM MMM     /^ ^\  ')
    print('  BBB    BBB   MMM  MMMM  MMM  AAAAAAAAAAA ---- EEEEE      MMM  MMMM  MMM    / 0 0 \ ')
    print('  BBB     BBB  MMM   MM   MMM  AAA     AAA      EEE        MMM   MM   MMM    V\ Y /V ')
    print('  BBB     BBB  MMM        MMM  AAA     AAA      EEE        MMM        MMM     / - \  ')
    print('  BBBBBBBBBB   MMM        MMM  AAA     AAA      EEEEEEEEE  MMM        MMM    /     | ')
    print('  BBBBBBBBB    MMM        MMM  AAA     AAA      EEEEEEEEE  MMM        MMM    V__) || ')
    print('  -----------------------------------------------------------------------            ')
    print('  © Jasper A. Vrugt, University of California Irvine & GPT-4 OpenAI''s language model')
    print('    ________________________________________________________________________')
    print('    Version 2.0, Dec. 2024, Beta-release: MATLAB implementation is benchmark')
    print('\n')

    n, K = D.shape              # Ensemble forecasts
    beta = np.ones(K) / K       # Initial weights
    Y = np.tile(y, (K, 1)).T    # Duplicate y model K times
    z_it = np.zeros((n, K))     # Initial latent variables
    ell_it = -np.inf
    err = np.ones((4))
    it = 0

    # Initialize variance treatment
    if options['VAR'] == '1':
        s = np.std(y) * np.random.rand()
    elif options['VAR'] == '2':
        s = np.std(y) * np.random.rand(K)
    elif options['VAR'] == '3':
        c = np.random.rand()/2
    elif options['VAR'] == '4':
        c = np.random.rand(K)/5

    while np.max(err) > errtol and it < maxit:
        ell = ell_it
        z = z_it

        if options['VAR'] in ['3', '4']:
            s = np.multiply(c, np.abs(D))

        # EXPECTATION STEP
        _, L = BMA_loglik(s, beta, Y, D, n, K, options, 0)
        z_it = np.multiply(beta, L)                         # Latent variable
        ell_it = np.sum(np.log(np.sum(z_it, axis=1)))       # Log-likelihood BMA model
        z_it = z_it / np.sum(z_it, axis=1, keepdims=True)   # Normalize latent variables

        # MAXIMIZATION STEP
        beta_it = np.sum(z_it, axis=0) / n  # New weights

        # Compute new sigma2's
        if options['VAR'] in ['1', '2']:
            if options['PDF'] == 'normal':
                s2_it = np.sum(z_it * (D - Y) ** 2, axis=0) / np.sum(z_it, axis=0)
                if options['VAR'] == '1':  # Constant group variance
                    s2_it = np.mean(s2_it) * np.ones(K)  # Copy mean value
            else:  # Nelder-Mead minimization
                # s2_it = minimize(lambda s: BMA_loglik(s, beta_it, Y, D, n, K, options, 1), s).x ** 2
                s2_it = minimize(BMA_loglik,s,args=(beta_it, Y, D, n, K, options, 1)).x ** 2
        elif options['VAR'] in ['3', '4']:
            # c_it = minimize(lambda c: BMA_loglik(c, beta_it, Y, D, n, K, options, 1), c)
            c_it = minimize(BMA_loglik,c,args=(beta_it, Y, D, n, K, options, 1))
            s2_it = np.multiply(c_it.x, np.abs(D)) ** 2  # matrix: c_t scalar or 1xK vector
            c = c_it.x  # use last value(s) of c_t

        # Convergence diagnostics
        err[0] = np.max(np.abs(beta_it - beta))  # Convergence weights
        err[1] = np.mean(np.max(np.abs(np.log(s2_it / s ** 2)), axis=0))  # Convergence variance(s)
        err[2] = np.max(np.abs(z - z_it))  # Convergence latent variables
        err[3] = np.max(np.abs(ell - ell_it))  # Convergence log-likelihood

        beta = beta_it
        s = np.sqrt(s2_it)  # Update weights/vars
        it += 1  # Iteration counter

    if options['VAR'] == '1':
        pass  # s = s * np.ones(K) (duplicate)
    elif options['VAR'] == '3':
        c = c * np.ones(K)

    # Return final results
    if options['VAR'] in ['1', '2']:
        return beta, s, ell_it, it
    elif options['VAR'] in ['3', '4']:
        return beta, c, ell_it, it
