function varargout = EM_bma(D,y,options,errtol,maxit)
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
%  SYNOPSIS: [beta,sigma,loglik,it] = EM_bma(D,y,options)                 %
%            [beta,sigma,loglik,it] = EM_bma(D,y,options,errtol)          %
%            [beta,sigma,loglik,it] = EM_bma(D,y,options,errtol,maxit)    %
%   where                                                                 %
%    D         [input] nxK matrix forecasts ensemble members              %
%    y         [input] nx1 vector with verifying observations             %
%    options   [input] structure BMA algorithmic variables                %
%     .PDF             string: conditional PDF for BMA method (MANUAL)    %
%                       = 'normal'     = normal distribution              %
%                       = 'lognormal'  = lognormal distribution           %
%                       = 'tnormal'    = truncated normal ( > 0)          %
%                       = 'gamma'      = gamma distribution               %
%     .VAR             string: variance treatment BMA method (MANUAL)     %
%                       =  '1'          = constant group variance         %
%                       =  '2'          = constant individual variance    %
%                       =  '3'          = nonconstant group variance      %
%                       =  '4'          = nonconstant ind. variance       %
%     .alpha           significance level for prediction limits BMA model %
%                       = [0.5 0.7 0.9 0.95 0.99]                         %
%   errtol    [input] OPTIONAL: error tolerance: default = 1e-4           %
%   maxit     [input] OPTIONAL: maximum # iterations: default = 10000     %
%   beta      [outpt] (1 x K)-vector of BMA weights                       %
%   sigma     [outpt] (1 x K)-vector of BMA standard deviations - or      %
%   c         [outpt] (1 x K)-vector of multipliers c: sigma = c*|D|      %
%   loglik    [outpt] BMA log-likelihood                                  %
%   it        [outpt] number of iterations to reach convergence           %
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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%  BUILT-IN CASE STUDIES                                                  %
%    Example 1   24-hour forecasts of river discharge                     %
%    Example 2   48-forecasts of sea surface temperature                  %
%    Example 3   48-forecasts of sea surface pressure                     %
%    Example 11  Forecasts of water levels                                %
%    Example 12  Hydrologic modeling                                      %
%    Example 13  Flood modeling                                           %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%  http://faculty.sites.uci.edu/jasper                                    %
%  http://faculty.sites.uci.edu/jasper/publications/                      %
%  https://scholar.google.com/citations?user=zkNXecUAAAAJ&hl=nl           %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

if nargin < 5, maxit = 1e4; end
if nargin < 4, errtol = 1e-4; end
if nargin < 3
    error(['EM_bma ERROR: TooFewInputs - Function EM_bma requires at ' ...
        'least three input arguments.']);
end
if ~sum(strcmp(options.PDF,{'normal','lognormal','tnormal','gamma'}))
    error(['EM_bma ERROR: Code only works if conditional PDF is a ' ...
        'normal, lognormal, truncated normal or gamma distribution']);
end
if ~sum(strcmp(options.VAR,{'1','2','3','4'}))
    error(['EM_bma ERROR: Code only works if variance treatment VAR is' ...
        'equal to ''1'', ''2'', ''3'' or ''4'' ']);
end
warning off                             % for fminsearch
[n,K] = size(D);                        % Ensemble forecasts
beta = ones(1,K)/K;                     % Initial weights
Y = repmat(y,1,K);                      % Duplicate y model K times
z_it = zeros(n,K);                      % Initial latent variables
ell_it = -inf; err = 1; it = 0;         % Constraints while loop

switch options.VAR                      % Variance treatment forecast PDF?
    case '1', s = std(y)*rand;
    case '2', s = std(y)*rand(1,K);
    case '3', c = 0.5*rand;
    case '4', c = 0.2*rand(1,K);
end

while (max(err) > errtol) && (it < maxit)   % Until ... do
    ell = ell_it; z = z_it;                     % Copy loglik and z
    switch options.VAR                          % Compute sigma
        case {'3','4'}, s = bsxfun(@times,c,abs(D));
    end
    %% EXPECTATION STEP
    [~,L] = BMA_loglik(s,beta,Y,D,n,K,options,0);   % Compute likelihood
    z_it = bsxfun(@times,beta,L);                   % Latent variable
    ell_it = sum(log(sum(z_it,2)));                 % Log-likelihood BMA model
    z_it = bsxfun(@rdivide,z_it,sum(z_it,2));       % Norm. latent variables
    %% MAXIMIZATION STEP
    beta_it = sum(z_it)/n;                          % New weights
    switch options.VAR                              % Compute new sigma2's
        case {'1','2'}
            switch options.PDF
                case 'normal'
                    s2_it = sum(z_it.*bsxfun(...        % sigma2 estimate
                        @minus,D,Y).^2)./sum(z_it);
                    if strcmp(options.VAR,'1')          % Comm. const. var.
                        s2_it = mean(s2_it)*ones(1,K);  % Copy mean value
                    end
                otherwise % Nelder-Mead simplex
                    s2_it = fminsearch(@(s) ...         % s2_it from minimization
                        BMA_loglik(s,beta_it,Y,...
                        D,n,K,options,1),s).^2;
            end
        case {'3','4'}
            c_it = fminsearch(@(c) BMA_loglik(c,...     % c_it from minimization     
                beta_it,Y,D,n,K,options,1),c);
            s2_it = bsxfun(@times,c_it,abs(D)).^2;      % matrix: c_t scalar or 1xK vector
            c = c_it;                                   % use last value(s) of c_t
    end
    %% CONVERGENCE DIAGNOSTICS
    err(1) = max(abs(beta_it - beta));              % Conv. weights
    err(2) = mean(max(abs(log(s2_it./s.^2))));      % Conv. variance(s)
    err(3) = max(max(abs(z - z_it)));               % Conv. latent variables
    err(4) = max(abs(ell - ell_it));                % Conv. log-likelihood
    beta = beta_it; s = sqrt(s2_it);                % update weights/vars
    it = it + 1;                                    % Iteration counter
end                                             % End while loop

if strcmp(options.VAR,'1')  % Duplicate K times sigma or multiplier c
    % s = s*ones(1,K); 
elseif strcmp(options.VAR,'3')
    c = c*ones(1,K);
end

switch options.VAR          % Group return argument
    case {'1','2'}, varargout = {beta , s , ell , it};
    case {'3','4'}, varargout = {beta , c , ell , it};
end

end
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% Secondary functions
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

% 1: BMA_loglik
function [ell,L] = BMA_loglik(x,beta,Y,D,n,K,options,flag)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%BMA_loglik: Computes the loglikelihood of the BMA model
%
% Written by Jasper A. Vrugt
% Los Alamos National Laboratory
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Make sure that weights and variances/multipliers do not go negative
if sum(any(x < 0))
    ell = inf; return
end
% Check implementation
if flag == 1 % fminsearch implementation
    switch options.VAR
        case '1' % constant: sigma = x (scalar)
            sigma = x*ones(n,K);
        case '2' % constant: sigma = x (vector)
            sigma = repmat(x,n,1);
        case '3' % nonconstant: c = x (scalar)
            sigma = x*abs(D);
        case '4' % nonconstant: c = x (vector c1...cK)
            sigma = repmat(x,n,1) .* abs(D);
    end
else % sigma is already known
    switch options.VAR
        case '1' % constant: sigma = x (scalar)
            sigma = x(1) * ones(n,K);
        case '2' % constant: sigma = x (vector)
            sigma = repmat(x,n,1); 
        otherwise
            sigma = x;
    end
end

% Compute likelihood
switch options.PDF
    case 'normal'
        L = exp(-1/2*((Y-D)./sigma).^2)./(sqrt(2*pi).*sigma);
    case 'lognormal'
        A = sigma.^2./(D.^2);
        S2 = log(A+1); S = sqrt(S2);
        mu = log(abs(D)) - S2/2;
        L = exp(-1/2*((log(Y) - mu)./S).^2) ./ ...
            (sqrt(2*pi).*Y.*S);
    case 'gamma'
        mu = abs(D); A = mu.^2./sigma.^2; B = sigma.^2./mu;
        Z = Y./B;
        U = (A-1).*log(Z) - Z - gammaln(A);
        L = exp(U)./B;
    case 'tnormal'
        mu = max(D,eps); a = 0; b = inf;
        L = tnormpdf(Y,mu,sigma,a,b);
end
% Return BMA log-likelihood
ell = - sum(log(L * beta'));

end
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
