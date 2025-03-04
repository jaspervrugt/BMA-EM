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

%% I received this data from someone who was using the MODELAVG toolbox. 
%% The data record is not long enough, but a fast practice for the 
%% different methods 

% Conditional distribution of ensemble members
options.PDF = 'tnormal';                % truncated normal conditional pdf
options.VAR = '3';                      % group nonconstant variance
options.alpha = [0.01 0.05 0.1 ...      % signf. levls prd intrvls BMA modl
    0.2 0.3 0.4 0.5];    

% Load ensemble forecasts & verifying data
W = load('water_levels.txt');           % daily discharge forecasts (mm/d) 
                                        % models and verifying data 
id_cal = 1:25;                          % indices of training period
D = W(id_cal,1:3); y = W(id_cal,4); 
[D,a,b] = Bias_correction(D,y);         % Linear bias correction

% Run BMA method
[beta,sigma,loglik,it] = ...            
    EM_bma(D,y,options);   
x = [beta,sigma];                       % Maximum lik. BMA weights and stds. 
[plimit,pdf_y,cdf_y] = ...              % Prediction limits & pdf & cdf @ y
    BMA_quantile(x,D,y,options);
mCRPS = mean(BMA_crps(x,D,y,options));  % CRPS BMA model: exact but slow
