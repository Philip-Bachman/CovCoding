function [ X_hat ] = covcode_decode( X, A, B )
% Get the outputs predicted from the inputs X, bases A, and basis weights B.
%
% Parameters:
%   X: input observations (obs_count x obs_dim)
%   A: basis matrices (obs_dim x obs_dim x basis_count)
%   B: basis weights (obs_count x basis_count)
%
% Output:
%   X_hat: the outputs predicted using the input parameters
%

obs_count = size(X,1);
obs_dim = size(X,2);
basis_count = size(A,3);

if (size(A,1) ~= obs_dim || size(A,2) ~= obs_dim)
    error('covcode_decode: mismatched basis/input dimensions\n');
end
if (size(B,2) ~= basis_count)
    error('covcode_decode: mismatched basis counts in A/B\n');
end
if (size(B,1) ~= obs_count)
    error('covcode_decode: mismatched obs counts in X/B\n');
end

X_hat = zeros(obs_count, obs_dim);
B_idx = abs(B) > 1e-4;
for i=1:basis_count,
    idx = B_idx(:,i);
    Xp = (squeeze(A(:,:,i)) * X(idx,:)')';
    X_hat(idx,:) = X_hat(idx,:) + bsxfun(@times, Xp, B(idx,i));
end

return

end

