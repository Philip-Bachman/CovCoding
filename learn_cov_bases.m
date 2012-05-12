function [ A ] = learn_cov_bases(...
    X, basis_count, code_spars, l1_bases, step, round_count, Ai )
% Learn a set of sparse gaussian graphical model bases for covariance coding,
% using the observations in X.
%
% Parameters:
%   X: input observations for self-regression (obs_count x obs_dim)
%   basis_count: number of basis matrices to learn (scalar)
%   code_spars:
%     if <  1: desired non-zero fraction of covariance codes (use glmnet)
%     if >= 1: desired number of non-zero code coefficients (use OMP)
%   l1_bases: l1 penalty to use for basis entries (scalar)
%   step: initial step size for gradient descent (scalar)
%   round_count: number of update rounds to perform (scalar)
%   Ai: (optional) set of starting bases (obs_dim x obs_dim x basis_count)
%
% Outputs:
%   A: learned covariance code bases
%

obs_dim = size(X,2);
if ~exist('Ai','var')
    A = rand_sggm_bases(obs_dim, basis_count);
else
    A = Ai(:,:,:);
end
if (code_spars >= 1)
    code_spars = round(code_spars);
end

do_cv = 0; % Whether to use hold-out set for line search in descent
nz_lvl = 0.1; % Noise level to add in gradient computations
% Learning loop
fprintf('Performing basis updates:\n');
for i=1:round_count,
    % Encode the input sequence using basis-projected sparse self-regression
    if (code_spars < 1)
        % Use glmnet/l1-regularization for encoding
        beta = covcode_encode(X, A, code_spars);
    else
        % Use OMP for encoding
        beta = covcode_encode(X, A, 0, code_spars);
    end
    % Update the bases using the computed encoding coefficients
    [ A_t post_err pre_err best_step ] = ...
        update_cov_bases(A, beta, X, step, l1_bases, nz_lvl, do_cv);
    fprintf('    round: %d, pre_err: %.4f post_err: %.4f, step: %.4f, kurt: %.4f\n',...
        i, pre_err, post_err, best_step, kurtosis(A_t(:)));
    A = A_t(:,:,:);
    step = best_step * 1.2;
    nz_lvl = nz_lvl * 0.95;
end


return

end

function [ A ] = rand_sggm_bases(obs_dim, basis_count)
% Create a set of random sparse ggm bases (maybe not sparse)
%

A = randn(obs_dim, obs_dim, basis_count);
for i=1:basis_count,
    for j=1:obs_dim,
        A(j,j,i) = 0;
    end
    A(:,:,i) = squeeze(A(:,:,i)) + transpose(squeeze(A(:,:,i)));
    A(:,:,i) = A(:,:,i) ./ std2(squeeze(A(:,:,i)));
end

return

end
