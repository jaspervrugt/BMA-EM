# ####################################################################### #
#                                                                         #
#   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE      222222   #
#   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE          22 22    #
#   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE         22     #
#   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE           22      #
#   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE      222222   #
#                                                                         #
# ####################################################################### #

# CASE STUDY II: 48-FORECASTS OF SEA SURFACE TEMPERATURE
# CHECK: A.E. RAFTERY ET AL., MWR, 133, pp. 1155-1174, 2005.

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
    'PDF': 'normal',                # Normal conditional pdf
    'VAR': '2',                     # Individual constant variance
    'alpha': [0.01, 0.05, 0.1, 0.5]  # Significance levels for prediction intervals and BMA model
}

# NOW LOAD DATA
T = np.loadtxt('temp_data.txt')  # 48-hour forecasts temperature (Kelvin) and verifying data 

# DEFINE ENSEMBLE AND VECTOR OF VERIFYING OBSERVATIONS (APRIL 16 TO JUNE 9, 2000)
start_idx = np.where((T[:, 0] == 2000) & (T[:, 1] == 4) & (T[:, 2] == 16))[0][0]
end_idx = np.where((T[:, 0] == 2000) & (T[:, 1] == 6) & (T[:, 2] == 9))[0][0]
D = T[start_idx:end_idx + 1, 4:9]  # Ensemble data (columns 5-9)
y = T[start_idx:end_idx + 1, 3]    # Verifying data (column 4)

# APPLY LINEAR BIAS CORRECTION TO ENSEMBLE (UP TO USER)
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
