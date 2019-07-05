%========================================================================
% 26/Jun/2019: KU Leuven, Alexander Gruber, Gabrielle De Lannoy
%              Initial version
%========================================================================

function [ stats ] = TCA_metrics( in_df, ref_col, select_col,...
                                  alpha_CI, bootstraps )

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data
%         ref_col(int)    : column index of reference data set for 
%                           estimating scaling coefficients [-] 
%         alpha_CI(float) : [0-1], 
%                           e.g. alpha=0.05 for 0.95 confidence interval
%         bootstraps(int) : optional number of bootstrap resamples that 
%                           should be calculated
% OUTPUT: stats[metric](select_col): resulting metrics w/ CI interval for 
%                           given data frame: R2_lower, R2_median, R2_upper
%                           ubRMSE_lower, ubRMSE_median, ubRMSE_upper
%                           beta_lower, beta_median, beta_upper
%
% Calculates triple collocation based squared correlations against the 
% unknown truth, SCALED error standard deviations, 
% and scaling coefficients for a given data frame.
%-------------------------------------------------------------------------

% only retain selected datasets for evaluation
in_df = in_df(:,[1 select_col]);
ref_col = find(select_col==ref_col);

if ~exist('bootstraps','var')
  bootstraps = 1000;
end

% initialize bootstrap generator to draw resamples from
% return in_df with regular time lags, without time column

[in_df, bss_ind, N_blocks, bl] = bootstrap(in_df);
N_df = length(in_df(:,1));

% stats stores results with p = direct TCA estimates,
% m = median of the bootstrap sampling distribution.

% calculate TCA metrics from the original sample
[stats.r2_p, stats.ubRMSE_p, stats.beta_p] = TCA_calc(in_df,ref_col);

arr  = NaN+zeros(3,3,bootstraps); %3: triple datasets, 3: metrics

% calculate TCA metrics from the bootstrap samples

for i=1:bootstraps
    % collect random indices to blocks, or simply resample if bl=1
    if bl>1
      tmp_ind = round((N_blocks-1)*rand(ceil(2*N_df/bl),1))+1;
    else
      tmp_ind = round((N_df-1)*rand(N_df,1))+1;  
    end
    % get the bootstrap sample
    ind = [bss_ind{tmp_ind}];
    ind = ind(1:N_df);
    bss = in_df(ind,:);
    [arr(:,1,i), arr(:,2,i), arr(:,3,i)] = TCA_calc(bss,ref_col);
end

ci_l = alpha_CI/2.*100;
ci_u = 100 - ci_l;
ci = prctile(arr,[ci_l 50 ci_u],3);

metric = {'r2','ubRMSE','beta'};
perc = {'l','m','u'};
for m=1:3
    for p=1:3
      cmd = ['stats.',metric{m},'_',perc{p},'=squeeze(ci(:,',...
              num2str(m),',',num2str(p),')'');'];
      eval(cmd);
    end
end

%========================================================================

function [in_df, bss_ind, N_blocks, bl] = bootstrap(in_df, bl)

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data
%         bl(sets)        : optional block length [-] 
%
% OUTPUT: in_df(time,sets): regularly lagged input data frame,
%                           without date/time column
%         bss_ind{N_blocks}: output structure with bootstrap sample
%                           indices per block
%         N_blocks(int)   : number of valid data blocks [-]
%         bl(int)         : average block lengths [lags]
%
% Bootstrap sample generator for a data frame with given block length [lags]
%-------------------------------------------------------------------------

% make sure that in_df has regular time intervals
if length(unique(in_df(1:end-1,1)-in_df(2:end,1))) > 1
  %in_df = regular_lag(in_df); 
  error('Error: collocate the data in time with regular time intervals');
end

% double check cross-masking

in_df = in_df(:,2:end); %exclude time column

if ~exist('bl','var')
    bl = calc_bootstrap_blocklength(in_df);
end

N_df = length(in_df(:,1));

N_blocks = 0;
i=1;
if bl>1
  % compose a structure of blocks with valid data (varying lengths) 
  while i<(N_df-bl/2)
    bl_ind = i:min(i+bl-1,N_df);
    bl_ind = bl_ind(~any(isnan(in_df(bl_ind,:)),2));
    if length(bl_ind)>bl/2
      N_blocks = N_blocks+1;
      bss_ind{N_blocks} = bl_ind; 
    end
    i=i+1;
  end
else
  % simply resample original
  bss_ind = 1:N_df;  
end

%========================================================================

function bl = calc_bootstrap_blocklength(in_df, rho)

% INPUT:  in_df(time,sets): regularly lagged input data frame (2D matrix),
%                           without date/time column
%         rho(sets)       : optional autocorrelation at lag-1 [-] 
%
% OUTPUT: bl(sets)        : block length [lags]
%
% Calculate optimal block length [lags] for block-bootstrapping of a data frame
%-------------------------------------------------------------------------

if ~exist('rho','var')
    rho = estimate_lag1_autocorr(in_df);
end

% Eq. 21 in Gruber et al. (2019)
bl = min([round((sqrt(6*length(in_df(:,1))).*rho./ (1 - rho.^2))^(2/3.)),...
          round(0.8*length(in_df(:,1)))]);

%========================================================================

function [R2, ubRMSE, beta] = TCA_calc(in_df, ref_col)

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; without time (3 data sets)
%          ref_col(scalar): column index of reference data set for 
%                           estimating scaling coefficients [-] 
% OUTPUT: [R2, ubRMSE, beta](sets): metrics for given data frame
%
% Calculates triple collocation based squared correlations against the 
% unknown truth, SCALED error standard deviations, 
% and scaling coefficients for a given data frame.
%-------------------------------------------------------------------------

C = cov(in_df);

ind = [1 2 3 1 2 3];
tmp = 1:3;

for i=1:3
  no_ref_ind = min(tmp(~(i==tmp | ref_col==tmp))); 
  snr(i) = abs((C(i,i)*C(ind(i+1),ind(i+2))/...
               (C(i,ind(i+1))*C(i,ind(i+2))))-1).^(-1);
  err_var(i) = abs( C(i,i) - ...
           C(i,ind(i+1))*C(i,ind(i+2))./C(ind(i+1),ind(i+2)));
  beta(i) = abs(C(ref_col,no_ref_ind)./C(i,no_ref_ind));
end

R2 = 1./(1 + snr.^(-1));
ubRMSE = sqrt(err_var).*beta;
%========================================================================

