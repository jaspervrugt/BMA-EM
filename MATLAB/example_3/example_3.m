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

%% CASE STUDY III: 48-FORECASTS OF SEA SURFACE PRESSURE
%% CHECK: A.E. RAFTERY ET AL., MWR, 133, pp. 1155-1174, 2005.

% Conditional distribution of ensemble members
options.PDF = 'gamma';                  % gamma conditional pdf
options.VAR = '1';                      % group constant variance
options.alpha = [0.01 0.05 0.1];        % signf. levls prd intrvls BMA modl

% Load ensemble forecasts & verifying data
P = load('pressure_data.txt');          % 48-h forcasts air-pressure (mbar) 
                                        % and verifying data
% April 16 to June 9, 2000
id = find(P(:,1) == 2000 & P(:,2) == 4 & P(:,3) == 16); start_id = id(1);
id = find(P(:,1) == 2000 & P(:,2) == 6 & P(:,3) == 9); end_id = id(end);
D = P(start_id:end_id,5:9); y = P(start_id:end_id,4);
[D,a,b] = Bias_correction(D,y);         % Linear bias correction

% Run BMA method
[beta,sigma,loglik,it] = ...            
    EM_bma(D,y,options);   
x = [beta,sigma];                       % Maximum lik. BMA weights and stds. 
[plimit,pdf_y,cdf_y] = ...              % Prediction limits & pdf & cdf @ y
    BMA_quantile(x,D,y,options);
mCRPS = mean(BMA_crps(x,D,y,options));  % CRPS BMA model: exact but slow
