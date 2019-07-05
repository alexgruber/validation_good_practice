%========================================================================
% 26/Jun/2019: KU Leuven, Alexander Gruber, Gabrielle De Lannoy
%              Initial version
%========================================================================

function rho = estimate_lag1_autocorr(in_df,tau)

% INPUT:  in_df(time,sets): regularly lagged (in time) input data frame
%                           (2D matrix)
%         tau(sets)       : optional tau [data lags]      
% OUTPUT: rho(float)      : combined autocorrelation at lag-1 [-] 
%
% Estimate geometric average median lag-1 auto-correlation
%-------------------------------------------------------------------------

if ~exist('tau','var')
    tau = estimate_tau(in_df);
end

% Calculate geometric average lag-1 auto-corr
% Eq. 19-20, and text below Eq. 21 in Gruber et al. (2019)
rho = exp(-1./tau);
rho = prod(rho.^(1/length(rho)));

%========================================================================

function tau = estimate_tau(in_df)

% INPUT:  in_df(time,sets): regularly lagged (in time) input data frame
%                           (2D matrix)
% OUTPUT: tau(sets)       : autocorrelation length [data lags] 
%
% Estimate characteristic time lengths for data columns
% by fitting an exponential auto-correlation function
%-------------------------------------------------------------------------

n_lags = 90; % parameter, maximum allowed lag size to be considered
n_cols = length(in_df(1,:)); 

rho = NaN + zeros(n_cols,n_lags);
tau = NaN + zeros(n_cols,1);

for i = 1:n_cols  
    
  Ser_l    =  in_df(:,i);  
  
  % calculate auto-correlation coefficients for different lags
  nn_lags  = min(n_lags,length(Ser_l));
  if isempty(isnan(Ser_l))
    rho(i,:) = autocorr(Ser_l,nn_lags);
  else
    if length(isnan(Ser_l))< 0.5*length(Ser_l) 
      disp(['Warning: ',num2str(length(isnan(Ser_l))),' NaN values']);
    end
    for l = 1:nn_lags
      rho(i,l)=corr(Ser_l(l:end),Ser_l(1:end-l+1),'rows','complete');
    end 
  end
  
  % fit exponential function to auto-correlations and estimate tau
  x        = 0:nn_lags-1;
  beta0    = [1];
  modelfun = @(b,x)(exp(-x/b));
  tau(i)   = nlinfit(x,rho(i,1:nn_lags),modelfun,beta0);
  tau(i)   = max(tau(i),1); %conservative estimate
  
  if abs(tau(i))>n_lags || isempty(tau(i)) || isnan(tau(i))
      ind      = find(rho(i,:)<exp(-1));
      if ~isempty(ind)
        tau(i) = ind(1);
      else
        tau(i) = n_lags;  
      end      
  end
 
end

%========================================================================
