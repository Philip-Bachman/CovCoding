function [ W ] = compute_whitener( X )
% Compute a ZCA whitening transform for the observations in the rows of X.

obs_dim = size(X,2);

H = cov(X);
[V D] = eigs(H, obs_dim);

D = D + (1e-6 * eye(obs_dim));
W = V * D^(-1/2) * V';

return

end

