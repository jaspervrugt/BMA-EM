% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%                  BBBBBBBBB    MMM     MMM      AAAAA                    %
%                  BBBBBBBBBB   MMM     MMM     AAAAAAA                   %
%                  BBB     BBB  MMMM   MMMM    AAA   AAA                  %
%                  BBB     BBB  MMMMM MMMMM    AAA   AAA                  %
%                  BBB    BBB   MMMMMMMMMMM    AAAAAAAAA                  %
%                  BBB    BBB   MMMMMMMMMMM    AAAAAAAAA                  %
%                  BBB     BBB  MMM     MMM    AAA   AAA                  %
%                  BBB     BBB  MMM     MMM    AAA   AAA                  %
%                  BBBBBBBBBB   MMM     MMM   AAA     AAA                 %
%                  BBBBBBBBB    MMM     MMM   AAA     AAA                 %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%  Ensemble Bayesian model averaging provides a methodology to explicitly %
%  handle conceptual model uncertainty in the interpretation and analysis %
%  of environmental systems. This method combines the predictive          %
%  capabilities of multiple different models and jointly assesses their   %
%  uncertainty. The probability density function (pdf) of the quantity of %
%  interest predicted by Bayesian model averaging is essentially a        %
%  weighted average of individual pdf's predicted by a set of different   %
%  models that are centered around their forecasts. The weights assigned  %
%  to each of the models reflect their contribution to the forecast skill %
%  over the training period                                               %
%                                                                         %
%  This code uses the Expectation Maximization algorithm for BMA model    %
%  training. This version for BMA model training assumes the predictive   %
%  PDF's of the ensemble members to equal a normal, lognormal, truncated  %
%  normal or gamma distribution with constant or nonconstant variance.    %
%  Please refer to the MODELAVG Package for a larger suite of predictive  %
%  PDF's of the ensemble members and other added functionalities & visual %
%  output to the screen                                                   %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%  LITERATURE                                                             %
%   Vrugt, J.A. (2024), Distribution-Based Model Evaluation and           %
%       Diagnostics: Elicitability, Propriety, and Scoring Rules for      %
%       Hydrograph Functionals, Water Resources Research, 60,             %
%       e2023WR036710,                                                    %
%           https://doi.org/10.1029/2023WR036710                          %
%   Vrugt, J.A. (2015), Markov chain Monte Carlo simulation using the     %
%       DREAM software package: Theory, concepts, and MATLAB              %
%       implementation, Environmental Modeling and Software, 75,          %
%       pp. 273-316                                                       %
%   Diks, C.G.H., and J.A. Vrugt (2010), Comparison of point forecast     %
%       accuracy of model averaging methods in hydrologic applications,   %
%       Stochastic Environmental Research and Risk Assessment, 24(6),     %
%       809-820, https://doi.org/10.1007/s00477-010-0378-z                %
%   Vrugt, J.A., C.G.H. Diks, and M.P. Clark (2008), Ensemble Bayesian    %
%       model averaging using Markov chain Monte Carlo sampling,          %
%       Environmental Fluid Mechanics, 8(5-6), 579-595,                   %
%           https://doi.org/10.1007/s10652-008-9106-3                     %
%   Wöhling, T., and J.A. Vrugt (2008), Combining multi-objective         %
%       optimization and Bayesian model averaging to calibrate forecast   %
%       ensembles of soil hydraulic models, Water Resources Research,     %
%       44, W12432,                                                       %
%          https://doi.org/10.1029/2008WR007154                           %
%   Vrugt, J.A., and B.A. Robinson (2007), Treatment of uncertainty using %
%       ensemble methods: Comparison of sequential data assimilation and  %
%       Bayesian model averaging, Water Resources Research, 43, W01411,   %
%           https://doi.org/10.1029/2005WR004838                          %
%   Vrugt, J.A., M.P. Clark, C.G.H. Diks, Q. Duan, and B.A.               %
%       Robinson (2006), Multi-objective calibration of forecast          %
%       ensembles using Bayesian Model Averaging, Geophysical Research    %
%       Letters, 33, L19817,                                              %
%           https://doi.org/10.1029/2006GL027126                          %
%   Raftery, A.E., T. Gneiting, F. Balabdaoui, and M. Polakowski (2005),  %
%       Using Bayesian model averaging to calibrate forecast ensembles,   %
%       Monthly Weather Revivew, 133, 1155–1174                           %
%   Raftery, A.E., and Y. Zheng (2003), Long-run performance of Bayesian  %
%       model averaging, Journal of the American Statistical Association, %
%       98, 931–938                                                       %
%   Raftery, A.E., D. Madigan, and J.A. Hoeting (1997), Bayesian model    %
%       averaging for linear regression models, Journal of the American   %
%       Statistical Association, 92, 179–191                              %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %
%                                                                         %
%  COPYRIGHT (c) 2015  the author                                         %
%                                                                         %
%   This program is free software: you can modify it under the terms      %
%   of the GNU General Public License as published by the Free Software   %
%   Foundation, either version 3 of the License, or (at your option)      %
%   any later version                                                     %
%                                                                         %
%   This program is distributed in the hope that it will be useful, but   %
%   WITHOUT ANY WARRANTY; without even the implied warranty of            %
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      %
%                                                                         %
%  MATLAB CODE                                                            %
%  © Written by Jasper A. Vrugt                                           %
%    Center for NonLinear Studies                                         %
%    Los Alamos National Laboratory                                       %
%    University of California Irvine                                      %
%  Version 0.5    July 2006                                               %
%  Version 1.0    January 2008  -- code cleaned and added examples        %
%  Version 1.1    July 2014     -- added quantiles                        %
%  Version 2.0    July 2024     -- added many more functionalities        %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %

%<><><><><><><><><><><><><><> Initialization <><><><><><><><><><><><><><><>

load data.txt;  % --> individual predictors
load Y.txt;     % --> corresponding observations

id = find(data < 1e-6); data(id) = 1e-6;    % All data at least 1e-6
[n,K] = size(data);                         % # members and # forecasts 

                % Define calibration and validation set (X are forecasts; Y = observations)
StartT = 1; EndT = 3000; X = data(StartT:EndT,1:K); Y = Y(StartT:EndT,1);

% How many data points do we use?
N = size(Y,1);

% Do we do a linear bias correction of the predictors or not?
reg.adjust = 'TRUE';

% Do linear bias correction
[A,B,Xcor] = ComputeAB(X,Y,reg);

% Define algorithmic parameters Expectation Maximization Algorithm
eps = 1e-5; maxiter = 25000;

% After EM optimization further tune sigma by optimization of CRPS score?
MIN.CRPS = 'FALSE'; 

% Constant variance for individual forecasts or model specific variances?
const.var = 'FALSE'; 

% ---------------------------------------------------------------------------------------------

% Optimize weights and variances with Expectation-Maximization algorithm 
[loglik,w,sigma,z,niter] = EM_normal(Xcor,Y,A,B,eps,maxiter,const,reg,MIN);

% Now define the uncertainty intervals
alpha = [0.005 0.05 0.50 0.95 0.995 ];

% Compute the associate prediction limits
[BMA_pred] = BMA_quantile(Xcor,sigma,w,alpha);
