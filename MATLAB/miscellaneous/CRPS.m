function crps = CRPS(beta,sigma,D,y)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% Computes the continues ranked probability score: Fast and approximate %%
%% for a normal predictive PDF with constant group/individual variances  %%
%% Note: the CRPS is treated as negative oriented: smaller is better     %%
%% This function is OBSOLETE, use BMA_quantile instead. This uses an     %%
%% exact quantile solution to the CRPS using built-in integral function  %%
%%                                                                       %%
%% SYNOPSIS: crps = CRPS(beta,sigma,D,y)                                 %%
%%  where                                                                %%
%%   beta      [input] (1 x K)-vector of BMA weights                     %%
%%   sigma     [input] (1 x K)-vector of BMA standard deviations - or    %%
%%   D         [input]  REQUIRED: n x K matrix of ensemble forecasts     %%
%%   y         [input]  REQUIRED: n x 1 vector of verifying observations %%
%%   crps      [outpt] Continuous Ranked Probability Score for BMA model %%
%%                                                                       %%
%% (c) Written by Jasper A. Vrugt, Feb 2006                              %%
%% Los Alamos National Laboratory                                        %%
%%                                                                       %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%

% Determine how many models 
[n,K] = size(D); 
% Set up the variance 
if length(sigma)==1
    s2 = sigma^2 * ones(1,K);
else
    s2 = sigma.^2;
end
% Compute the standard deviation
s = sqrt(s2);

% Now begin computing the first term in the CRPS formula.  
% This is a double sum since it is over w(i)*w(j) for all i and j
crps = zeros(1,n); t = 1;
while t <= n
    % Set some initial values
    firstSum = 0; secondSum = 0;
    % First get the first sum (the double sum of the CRPS)
	k = 1;
    while k <= K
          j = 1;
          while j <= K
                % Compute total variance
                ts2 = s2(k) + s2(j);
                % Compute total standard deviation
                ts = sqrt(ts2);
                % Compute mean
                dD = D(t,k) - D(t,j);
                % Compute first sum
                firstSum = firstSum + beta(k) * beta(j) * absexp(dD,ts);
                j = j + 1;
          end
          k = k + 1;
    end
	% Now get the second sum.  This one is only over all w(i) for all i.
    k = 1;
    while k <= K
        % Total variance
        ts2 = s2(k);
        % Total standard deviation
        ts = sqrt(ts2);
        % Mean
        dy = D(t,k) - y(t);
        % Compute Second sum
        secondSum = secondSum + beta(k) * absexp(dy,ts);
        % Update i
        k = k +1;
    end
    % Szekely's expression for CRPS: first and second sum equals the CRPS
    crps(t)  = -0.5*firstSum + secondSum;
    % Update the k counter
    t = t + 1;
end
% Now determine output variable
crps = abs(mean(crps));

end
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% Secondary functions
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

% 1: absexp
function y = absexp(mu,sigma)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% This function computes the absolute exp given mu and sigma: for CRPS    %
%                                                                         %
% SYNOPSIS: y = absexp(mu,sigma)                                          %
%  where                                                                  %
%   mu        [input] mean of the normal distribution                     %
%   sigma     [input] standard deviation of the normal distribution       %
%   y         [outpt] scalar                                              %
%                                                                         %
% (c) Written by Jasper A. Vrugt, Feb 2006                                %
% Los Alamos National Laboratory                                          %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

y = (1/sqrt(pi)) * sigma * sqrt(2) * exp((-.5*mu^2)/(sigma^2)) ...
        + mu * erf(.5*mu*sqrt(2)/sigma);

end
