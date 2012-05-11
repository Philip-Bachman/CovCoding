function [ A_grads ] = basis_gradients( A, B, X, l1_pen )
% Get the element-wise gradients of bases for:
%   ||x - sum_i(b(i)*A(i)*x)||^2 + l1_pen * sum_i ||A(i)||_1
% for each pair of rows b/x in B/X.
%
% Parameters:
%   A: the bases used in self-regression (obs_dim x obs_dim x basis_count)
%   B: self-regression coefficients for bases in A (obs_count x basis_count)
%   X: the input observations for self-regression (obs_count x obs_dim)
%   l1_pen: l1 sparsifying penalty to place on basis entries (scalar, 1 x 1)
%
% Outputs:
%   A_grads: partial derivatives of the objective with respect to the elements
%            of each basis
%

if ~exist('l1_pen','var')
    l1_pen = 0.0;
end

obs_dim = size(X,2);
obs_count = size(X,1);
A_grads = zeros(size(A));

% Compute basis gradients when each basis is a matrix
if (obs_dim ~= size(A,1) || obs_dim ~= size(A,2))
    error('basis_gradients: dims of bases and observations do not match.\n');
end

X_hat = covcode_decode(X, A, B);
X_res = X_hat - X;
Bnz = abs(B) > 1e-4;
for obs_num=1:obs_count,
    bnz_idx = find(Bnz(obs_num,:));
    part_grads = X_res(obs_num,:)' * X(obs_num,:);
    part_grads = part_grads .* 2.0;
    for i=bnz_idx,
        A_grads(:,:,i) = A_grads(:,:,i) + (part_grads .* B(obs_num,i));
    end
end
A_grads = A_grads ./ obs_count;

% Add soft-absolute regularization, to sparsify the learned basis matrices.
A_grads = A_grads + ((A ./ sqrt(A.^2 + 1e-5)) .* l1_pen);

return

end



