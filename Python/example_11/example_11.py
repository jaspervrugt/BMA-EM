# ####################################################################### #
#                                                                         #
# EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE     1111   1111 #
# EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE        11 11  11 11 #
# EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE    11  11 11  11 #
# EE       XXXX   AAAAAA  MM   MM  PP      LL      EE           11     11 #
# EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE       11     11 #
#                                                                         #
# ####################################################################### #

# CASE STUDY XI: MODEL AVERAGING WITH WATER LEVELS

# I received this data from someone who was using the MODELAVG toolbox. 
# The data record is not long enough, but a fast practice for the 
# different methods 

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
from scipy.stats import norm
import matplotlib.pyplot as plt
from BMA_EM import BMA_EM
from BMA_EM_functions import Bias_correction, BMA_quantile, BMA_crps, fast_CRPS 

# Conditional distribution of ensemble members
options = {
    'PDF': 'normal',                # Normal conditional pdf
    'VAR': '1',                     # Group nonconstant variance
    'alpha': [0.01, 0.05, 0.1, 0.5]  # Significance levels for prediction intervals and BMA model
}

# NOW LOAD DATA
W = np.loadtxt('water_levels.txt')  # Daily discharge forecasts (mm/day) and verifying data
idx_cal = slice(0, 25)              # Indices of training period (1-25 in MATLAB corresponds to 0-24 in Python)
idx_eval = slice(25, W.shape[0])    # Indices of evaluation data period

# DEFINE TRAINING ENSEMBLE AND VECTOR OF VERIFYING OBSERVATIONS
D_cal = W[idx_cal, 0:3]  # Ensemble (models 1-3)
y_cal = W[idx_cal, 3]    # Verifying data (column 4)

# APPLY LINEAR BIAS CORRECTION TO ENSEMBLE (UP TO USER)
D, a, b = Bias_correction(D_cal, y_cal)

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

