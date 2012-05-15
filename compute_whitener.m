function [ W ] = compute_whitener( X )
% Compute a ZCA whitening transform for the observations in the rows of X.
%
% Parameters:
%   X: (row-wise) data for which to compute whitening transform
% Outputs:
%   W: struct containing data mean (in W.M) and whitener (in W.W)
%
% Note: to whiten future row-wise data, do "X = bsxfun(@minus, X, W.M) * W.W';"
%
obs_dim = size(X,2);
% Compute data covariance and eigenvalues/eigenvectors of data covariance
H = cov(X);
[V D] = eig(H);
% Regularize matrix of eigenvalues, to avoid issues with small/zero values
D = D + (1e-6 * eye(obs_dim));
% Compute data mean and whitener
W = struct();
W.W = V * D^(-1/2) * V';
W.M = mean(X);
return
end

