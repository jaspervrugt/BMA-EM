% ----------------------------------------------------------------------- %
%  MM   MM   OOOOO   DDDDDD   EEEEEEE  LL         AAA    VV   VV   GGGGG  %
%  MM   MM  OOOOOOO  DDDDDDD  EEEEEEE  LL        AA AA   VV   VV  GG   GG %
%  MMM MMM  OO   OO  DD   DD  EE       LL        AA AA   VV   VV  GG   GG %
%  MM M MM  OO   OO  DD   DD  EEEEE    LL       AA   AA  VV   VV  GGGGGGG %
%  MM   MM  OO   OO  DD   DD  EEEEE    LL       AAAAAAA  VV   VV   GGGGGG %
%  MM   MM  OO   OO  DD   DD  EE       LL       AA   AA   VV VV        GG %
%  MM   MM  OOOOOOO  DDDDDDD  EEEEEEE  LLLLLLL  AA   AA    VVV     GGGGGG %
%  MM   MM   OOOOO   DDDDDD   EEEEEEE  LLLLLLL  AA   AA     V     GGGGGGG %
% ----------------------------------------------------------------------- %
%                                                                         %
%  EXAMPLE 1: Ensemble of discharge models                                %
%   Vrugt, J.A., and B.A. Robinson (2007), Treatment of uncertainty using %
%       ensemble methods: Comparison of sequential data assimilation and  %
%       Bayesian model averaging, Water Resources Research, 43, W01411,   %
%           https://doi.org/10.1029/2005WR004838                          %

% Conditional distribution of ensemble members
options.PDF = 'normal';                 % normal conditional pdf
options.VAR = '3';                      % group nonconstant variance
options.alpha = [0.01 0.05 0.1 0.5];    % signf. levls prd intrvls BMA modl

% Load ensemble forecasts & verifying data
S = load('discharge_data.txt');         % daily discharge forecasts (mm/d) 
                                        % models and verifying data 
id_cal = 1:3000;                        % start/end training period
D = S(id_cal,1:8); y = S(id_cal,9);     % Ensemble forcsts & verifying data
[D,a,b] = Bias_correction(D,y);         % Linear bias correction ensemble forecasts

% Run BMA method
[beta,sigma,loglik,it] = ...            
    EM_bma(D,y,options);   
x = [beta,sigma];                       % Maximum lik. BMA weights and stds. 
[plimit,pdf_y,cdf_y] = ...              % Prediction limits & pdf & cdf @ y
    BMA_quantile(x,D,y,options);
mCRPSa = CRPS(beta,sigma,D,y);          % CRPS BMA model: fast & apprximate
mCRPSb = mean(BMA_crps(x,D,y,options)); % CRPS BMA model: exact but slow
