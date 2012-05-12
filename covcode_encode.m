function [ beta l2_reg ] = covcode_encode( X, A, sparsity, omp_num )
% Compute the covariance-codes for the observations in X, using the set of
% sparse self-regression bases in A. Encoding is performed in one of the
% following forms:
%
% Parameters:
%   X: observations for which to compute codes (obs_count x in_dim)
%   A: basis matrices for the codes (in_dim x in_dim x basis_count)
%   sparsity: the maximum allowable rate of non-zeros in code coefficients
%   omp_num: the number of bases to include for OMP encoding; if this parameter
%            is passed, then it is assumed that OMP encoding is desired.
%
% Output:
%   beta: the learned code coefficients for each observation in X
%   l2_reg: L2 regularization weight used in each code
%

if exist('omp_num','var')
    do_omp = 1;
    code_type = sprintf('OMP %d',omp_num);
else
    do_omp = 0;
    code_type = sprintf('glmnet %.2f',sparsity);
end

obs_count = size(X,1);
in_dim = size(X,2);
basis_count = size(A,3);
block_size = 50;

bases = cell(basis_count,1);
for i=1:basis_count,
    bases{i} = squeeze(A(:,:,i));
end

% Setup the options structure for glmnet
options = glmnetSet();
options.dfmax = max(10,ceil(sparsity * in_dim));
options.alpha = 0.9;

% Compute either the sparse or dense lwr for each time point
beta = zeros(obs_count, basis_count);
l2_reg = zeros(obs_count, 1);
fprintf('Computing cov-codes (%s):', code_type);
block_idx = 1;
block_num = 1;
x_p_block = zeros(block_size,in_dim,basis_count);
for obs_num=1:obs_count,
    if (mod(obs_num, round(obs_count/50)) == 0),
        fprintf('.');
    end
    if (block_idx == 1)
        block_start = ((block_num - 1) * block_size) + 1;
        block_end = block_start + block_size - 1;
        for b=1:basis_count,
            x_p_block(:,:,b) = (bases{b} * X(block_start:block_end,:)')';
        end
    end
    % Project the current observation onto the bases
    x = X(obs_num,:)';
    x_p = squeeze(x_p_block(block_idx,:,:));
    if (norm(x) > 1e-5)
        % Encode the observation in terms of the its projections onto the bases
        if (do_omp == 0)
            % Use glmnet for regression when a sparse fit is desired
            fit = glmnet(x_p, x, 'gaussian', options);
            beta_obs = fit.beta(:,end)';
            lambda_obs = fit.lambda(end);
            for j=2:numel(fit.lambda),
                if ((fit.df(j) / basis_count) > sparsity)
                    beta_obs = fit.beta(:,j-1)';
                    lambda_obs = fit.lambda(j-1);
                    break
                end
            end
            beta(obs_num,:) = beta_obs;
            l2_reg(obs_num) = lambda_obs * (1 - options.alpha);
        else
            % Use OMP for the encoding
            beta_obs = omp_encode( x', x_p, omp_num);
            beta(obs_num,:) = beta_obs;
            l2_reg(obs_num) = 0;
        end
    end
    if (block_idx < block_size)
        block_idx = block_idx + 1;
    else
        block_idx = 1;
        block_num = block_num + 1;
    end
end
fprintf('\n');

return

end


function [ B ] = omp_encode( X, A, omp_num )
% Use orthogonal matching pursuit to encode the observations in X using the
% columns of A as bases.
%
% Parameters:
%   X: observations to encode (obs_count x obs_dim)
%   A: bases with which to encode observations (obs_dim x basis_count)
%   omp_num: the number of bases to include in each reconstruction
% Outputs:
%   B: encoding of observations in terms of bases (obs_count x basis_count)
%
obs_count = size(X,1);
basis_count = size(A,2);
A_sqs = sum(A.^2);
Xr = X;
B = zeros(obs_count, basis_count);
B_idx = zeros(obs_count, omp_num);
for i=1:omp_num,
    dots = Xr * A;
    scores = dots ./ (sqrt(sum(Xr.^2,2)) * sqrt(A_sqs));
    [max_scores max_idx] = max(abs(scores),[],2);
    for j=1:obs_count,
        idx = max_idx(j);
        B_idx(j,i) = idx;
        w = (Xr(j,:) * A(:,idx)) / A_sqs(idx);
        Xr(j,:) = Xr(j,:) - (A(:,idx)' .* w);
        B(j,idx) = B(j,idx) + w;
    end
end
return
end

