# ####################################################################### #
#                                                                         #
#   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE        1111   #
#   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE           11 11   #
#   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE       11  11   #
#   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              11   #
#   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE          11   #
#                                                                         #
# ####################################################################### #

#  EXAMPLE 1: Ensemble of discharge models                                #
#   Vrugt, J.A., and B.A. Robinson (2007), Treatment of uncertainty using #
#       ensemble methods: Comparison of sequential data assimilation and  #
#       Bayesian model averaging, Water Resources Research, 43, W01411,   #
#           https://doi.org/10.1029/2005WR004838                          #

import sys
import os

# Get the current working directory
current_directory = os.getcwd()
# Go up one directory
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
# add this to path
sys.path.append(parent_directory)
# Add another directory
misc_directory = os.path.abspath(os.path.join(parent_directory, 'miscellaneous'))
# add this to path
sys.path.append(misc_directory)

import numpy as np
from BMA_EM import BMA_EM
from BMA_EM_functions import Bias_correction, BMA_quantile, BMA_crps, fast_CRPS 

# Conditional distribution of ensemble members
options = {
    'PDF': 'gamma',                  # Normal conditional pdf
    'VAR': '3',                      # Group nonconstant variance
    'alpha': [0.01, 0.05, 0.1, 0.5]} # Significance levels for prediction intervals and BMA model

# Load ensemble forecasts & verifying data
S = np.loadtxt('discharge_data.txt')  # Daily discharge forecasts (mm/d), models and verifying data

id_cal = range(3000)  # Start/end training period (indices)
D = S[id_cal, :8]     # Ensemble forecasts
y = S[id_cal, 8]      # Verifying data

# Bias correction
D, a, b = Bias_correction(D, y)

if __name__ == '__main__':
	# Run Bayesian Model Averaging (BMA)
	beta, sigma, loglik, it = BMA_EM(D, y, options)

	# Maximum likelihood BMA weights and standard deviations
	x = np.hstack([beta, sigma])

	# BMA prediction limits & pdf & cdf @ y
	plimit, pdf_y, cdf_y = BMA_quantile(x, D, y, options)

	# CRPS score of BMA model: fast & approximate for normal PDF
	mCRPS = fast_CRPS(beta, sigma, D, y)

	# CRPS score of BMA model: accurate
	CRPS = BMA_crps(x, D, y, options)

    # Print CRPS
	print("CRPS fast is equal to:", mCRPS)

    # Print CRPS
	print("CRPS exact is equal to:", np.mean(CRPS))
