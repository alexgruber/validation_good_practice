%========================================================================
% 26/Jun/2019: KU Leuven, Alexander Gruber, Gabrielle De Lannoy
%              Initial version
%========================================================================

function [ stats ] = relative_metrics( in_df, ...
    AC, c_or_p, ref_col, select_col, alpha_CI )

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data
%         AC(boolean)     : autocorrelation-corrected samples or not (1/0)
%         c_or_p(string)  : 'complete' or 'pairwise' crossmasking
%         ref_col(int)    : index to 1 reference dataset
%         select_col(int_array): index to (possibly multiple) selected
%                           dataset(s) that are to be evaluated
%         alpha_CI(float) : [0-1], 
%                           e.g. alpha=0.05 for 0.95 confidence interval
% OUTPUT: stats.[metric](select_col): structure with metric values for
%                           all selected datasets (select_col), 
%                           relative to one reference dataset (ref_col)
%
% Calculate relative metrics.
%-------------------------------------------------------------------------

% only retain selected datasets for evaluation
select_col = select_col(~(ref_col == select_col)); %exclude refdata

n_col_stat = length(select_col);

if (strcmp(c_or_p,'complete'))

    in_df = in_df(:,[1 ref_col select_col]); 
                            %1st col = date (needed to regularize data 
                            %          in time with equal lags),
                            %2nd col = refdata
    
    % (effective) number of pairs, identical for all 
    if AC>0
        stats.npairs(1:n_col_stat) = correct_n(in_df);
    else
        stats.npairs(1:n_col_stat) = size(in_df,1);
    end
    
    % reduce to cross-masked data, remove time column
    t = isnan(in_df); 
    in_df = in_df(~any(t,2),:); 
    in_df = in_df(:,2:end);     
    
    % bias metrics
    [stats.bias, stats.bias_l, stats.bias_u, ~] = ...
        bias(in_df,alpha_CI,stats.npairs);
    % ubRMSD metrics
    [stats.ubRMSD, stats.ubRMSD_l, stats.ubRMSD_u, ~] = ...
        ubRMSD(in_df,alpha_CI,stats.npairs);    
    % correlation metrics
    [stats.R, stats.R_l, stats.R_u, ~, stats.p] = ...
        Pearson_R(in_df,alpha_CI,stats.npairs);
    
else

    ii=0;
    % compute metrics for each data-pair separately
    for i=select_col
    
        ii=ii+1;  
        tmp_df = in_df(:,[1 ref_col i]);
        
        % (effective) number of pairs, different per dataset combination 
        if AC>0
            stats.npairs(ii) = correct_n(tmp_df);
        else
            stats.npairs(ii) = size(tmp_df,1);
        end 
        
        % reduce to cross-masked data, remove time column
        t1 = isnan(tmp_df(:,1));      
        t2 = isnan(tmp_df(:,2)); 
        t = t1;
        t((t2==1)) = 1;
        tmp_df = tmp_df(~any(t,2),:); %reduce in time
        tmp_df = tmp_df(:,2:end);    %remove time column
       
        % bias metrics
        [stats.bias(ii), stats.bias_l(ii), stats.bias_u(ii), ~] = ...
            bias(tmp_df,alpha_CI,stats.npairs(ii));
        % ubRMSD metrics
        [stats.ubRMSD(ii), stats.ubRMSD_l(ii), stats.ubRMSD_u(ii), ~] = ...
            ubRMSD(tmp_df,alpha_CI,stats.npairs(ii));    
        % correlation metrics
        [stats.R(ii), stats.R_l(ii), stats.R_u(ii), ~, stats.p(ii)] = ...
            Pearson_R(tmp_df,alpha_CI,stats.npairs(ii));

    end

end

%========================================================================

function n = correct_n(in_df, rho)

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data
%         rho(sets)       : optional autocorrelation at lag-1 [-]
%
% OUTPUT: n(sets)         : corrected sample size [-]
%
% Calculate corrected sample size based on averge lag-1 auto-correlation.
%-------------------------------------------------------------------------

if ~exist('rho','var')
    
    % make sure that in_df has regular time intervals
    if length(unique(in_df(1:end-1,1)-in_df(2:end,1))) > 1
      %in_df = regular_lag(in_df); 
      error('Error: collocate the data in time with regular time intervals');
    end
    
    rho   = estimate_lag1_autocorr(in_df(:,2:end));
end

% only retain valid data sample length
t = ~isnan(in_df(:,1));
in_df = in_df(t,:);
if any(isnan(in_df))
    error('ERROR: samples are not collocated in time');
end

% Eq. 18 in Gruber et al. (2019)
n = round(length(in_df(:,1)).*((1-rho)/(1+rho)));

%========================================================================

function [bias, CI_l, CI_u, n] = bias(in_df, alpha, n_corr)

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data, first col=reference
%         alpha(float)    : confidence level [0-1]
%         n_corr(sets)    : optional auto-correlation-corrected sample size 
%
% OUTPUT: bias(sets)      : temporal mean bias for all datasets
%         CI_l, CI_u(sets): lower and upper confidence levels 
%         n(sets)         : original or auto-correlation corrected sample size
%
% Calculates temporal mean biases and their confidence intervals 
% based on Student's t-distribution.
%-------------------------------------------------------------------------

if ~exist('n_corr','var') %simple sample size, not corrected for autocorrelation
    t = isnan(in_df); 
    in_df = in_df(~any(t,2),:);
    n = size(in_df,1);
else
    n = n_corr;
end

err = repmat(in_df(:,1),1,size(in_df,2)-1) - in_df(:,2:end);   
bias = nanmean(err,1);

% Eq. 14 in Gruber et al. (2019)
ser = nanstd(err,1) ./ sqrt(n);
crit = tinv((1 - alpha / 2), max(n-1,0)) .* ser;

CI_l = bias - crit;
CI_u = bias + crit;

%========================================================================

function [ubRMSD, CI_l, CI_u, n] = ubRMSD(in_df,alpha,n_corr)

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data, first col=reference
%         alpha(float)    : confidence level [0-1]
%         n_corr(sets)    : optional auto-correlation-corrected sample size 
%
% OUTPUT: ubRMSD(sets)    : temporal ubRMSD for all datasets
%         CI_l, CI_u(sets): lower and upper confidence levels 
%         n(sets)         : original or auto-correlation corrected sample size
%
% Calculates the Pearson correlation coefficient and its confidence intervals 
% based on Fischer's z-transformation.
%-------------------------------------------------------------------------

if ~exist('n_corr','var') %simple sample size, not corrected for autocorrelation
    t = isnan(in_df); 
    in_df = in_df(~any(t,2),:);
    n = size(in_df,1);
else
    n = n_corr;
end

err = repmat(in_df(:,1),1,size(in_df,2)-1) - in_df(:,2:end);   
ubRMSD = nanstd(err,1);

% Eq. 15 in Gruber et al. (2019)
crit_l = sqrt(chi2inv(1 - alpha/2, n-1))./sqrt(n-1);
crit_u = sqrt(chi2inv(alpha/2, n-1))./sqrt(n-1);

CI_l = ubRMSD ./ crit_l;  
CI_u = ubRMSD ./ crit_u;

%========================================================================

function [R, CI_l, CI_u, n, p] = Pearson_R(in_df, alpha, n_corr)

% INPUT:  in_df(time,sets): input data frame (2D matrix), regularly
%                           lagged or not; first column is time,
%                           next columns are data, first col=reference
%         alpha(float)    : confidence level [0-1]
%         n_corr(sets)    : optional auto-correlation-corrected sample size 
%
% OUTPUT: R (sets)        : temporal Pearson correlation for all datasets
%         CI_l, CI_u(sets): lower and upper confidence levels 
%         n(sets)         : original or auto-correlation corrected sample size
%         p (sets)        : p-value (significance)
%
% Calculates the Pearson correlation coefficient and its confidence intervals 
% based on Fischer's z-transformation.
%-------------------------------------------------------------------------

% by default, sample size is not corrected for autocorrelation
if ~exist('n_corr','var') 
    t = isnan(in_df); 
    in_df = in_df(~any(t,2),:);
    n = size(in_df,1);
else
    n = n_corr;
end

X = repmat(in_df(:,1),1,size(in_df,2)-1); %reference data
Y = in_df(:,2:end);

[R,p] = corr(X,Y);

% only keep correlation versus reference data
R = R(1,:);
p = p(1,:);

% Fisher's z-transform for confidence intervals, 
% Eq. 16 in Gruber et al. (2019)
z = 0.5 * log((1+R)./(1-R));

if n>3
    zalpha = (-erfinv(alpha-1)).*sqrt(2)./sqrt(n-3);
    % Eq. 17 in Gruber et al. (2019)
    CI_l = tanh(z-zalpha);
    CI_u = tanh(z+zalpha);
else
    CI_l = NaN;
    CI_u = NaN;
end
%========================================================================
