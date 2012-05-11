function [ beta l2_reg ] = covcode_encode( X, A, sparsity )
% Compute the covariance-codes for the observations in X, using the set of
% sparse self-regression bases in A.
%
% Parameters:
%   X: observations for which to compute codes (obs_count x in_dim)
%   A: basis matrices for the codes (in_dim x in_dim x basis_count)
%   sparsity: the maximum allowable rate of non-zeros in code coefficients
%
% Output:
%   beta: the learned code coefficients for each observation in X
%   l2_reg: L2 regularization weight used in each code
%

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
fprintf('Computing cov-codes:');
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
    % Perform the kernel-weighted, l1-regularized regression on observations
    if (sparsity < 0.99)
        % Do an L1-regularized regression when a sparse fit is desired
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
        % Do a simple linear regression when a dense solution is desired
        error('covcode_encode(): sparsity must be < 1.\n');
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

