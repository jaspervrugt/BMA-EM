# Bayesian Model Averaging - Expectation Maximization in MATLAB and Python

## Description

Ensemble Bayesian model averaging provides a methodology to explicitly handle conceptual model uncertainty in the interpretation and analysis of environmental systems. This method combines the predictive capabilities of multiple different models and jointly assesses their uncertainty. The probability density function (pdf) of the quantity of interest predicted by Bayesian model averaging is essentially a weighted average of individual pdf's predicted by a set of different models that are centered around their forecasts. The weights assigned to each of the models reflect their contribution to the forecast skill over the training period. This code uses the Expectation Maximization algorithm for BMA model training. This version for BMA model training assumes the predictive PDF's of the ensemble members to equal a normal, lognormal, truncated normal or gamma distribution with constant or nonconstant variance. Please refer to the MODELAVG Package for a larger suite of predictive PDF's of the ensemble members and other added functionalities & visual output to the screen.

## Getting Started

### Installing: MATLAB

* Download and unzip the zip file 'MATLAB_code_BMA_EM_V2.0.zip' in a directory 'BMA-EM'
* Add the toolbox to your MATLAB search path by running the script 'install_BMA_EM.m' available in the root directory
* You are ready to run the examples.

### Executing program

* After intalling, you can simply direct to each example folder and execute the local 'example_X.m' file.
* Please make sure you read carefully the instructions (i.e., green comments) in 'install_BMA_EM.m'  

### Installing: Python

* Download and unzip the zip file 'Python_code_BMA_EM_V2.0.zip' to a directory called 'BMA_EM'.

### Executing program

* Go to Command Prompt and directory of example_X in the root of 'BMA_EM'
* Now you can execute this example by typing 'python example_X.py'
* Instructions can be found in the file 'BMA_EM.py' 
  
## Authors

* Vrugt, Jasper A. (jasper@uci.edu) 

## Version History

* 1.0
    * Initial Release
* 2.0
    * Python implementation
    * New built-in case studies and conditional PDFs

## Acknowledgments
