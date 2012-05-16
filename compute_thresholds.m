function [ threshs ] = compute_thresholds( patches, A, thresh_rates, W )
% Compute thresholds that obtain the threshold rates given in thresh_rates.
% Thresholds are based on the statistics of the distribution of feature
% activations observed when "encoding" the patches in "patches" using the
% dictionary described by "A".
%
% Parameters:
%   patches: the patches with which to estimate activation distribution
%   A: the bases to use for copmuting feature activations
%   thresh_rates: the rates for which to determine appropriate thresholds
%   W: an optional whitening transform for preprocessing
% Outputs:
%   im_feats: covcode features for the image
%

patch_pix = size(patches,2);
% Get the number of bases and the type of features to compute
if (numel(size(A)) == 3)
    % 3D A means we are computing covariance features
    basis_count = size(A,3);
    feat_type = 1;
else
    if (numel(size(A)) == 2)
        % 2D A means we are computing linear features
        basis_count = size(A,2);
        feat_type = 2;
    else
        error('compute_thresholds: A should be either 2 or 3 dimensional.\n');
    end
end
% Check if a whitening transform was given for preprocessing
if ~exist('W','var')
    W.M = zeros(1,patch_pix);
    W.W = eye(patch_pix);
else
    if (size(W.W,1) ~= patch_pix || size(W.W,2) ~= patch_pix ||...
            size(W.M,2) ~= patch_pix)
        error('compute_thresholds: whitener has the wrong dimension\n');
    end
end
fprintf('Computing triangle activation thresholds{\n');
% Compute feature activations for each patch
patches = ZMUN(patches);
patches = bsxfun(@minus,patches,W.M) * W.W';
feat_acts = compute_features(patches,A,basis_count,feat_type);

% Sort feature activations, to facilitate threshold finding
feat_acts = sort(abs(feat_acts(:)),'ascend');
threshs = zeros(1,numel(thresh_rates));
for t=1:numel(thresh_rates),
    % Compute a suitable threshold for each desired threshold rate
    t_idx = thresh_rates(t) * numel(feat_acts);
    threshs(t) = (feat_acts(floor(t_idx)) + feat_acts(ceil(t_idx))) / 2;
end
fprintf('}\n');

return
end


function [ patch_feats ] = compute_features(...
    patches, A, basis_count, feat_type)
% Compute features for the given patches, given the bases in A.
%
% Parameters:
%   patches: patches for which to compute features (patch_count x patch_pix)
%   A: bases for features (patch_pix (x patch_pix) x basis_count)
%   basis_count: the number of bases in A
%   feat_type: whether to compute linear or covariance features (1 or 2)
% Outputs:
%   patch_feats: the features computed per-patch (patch_count x basis_count)
%
patch_feats = zeros(size(patches,1),basis_count);
patch_norms = sqrt(sum(patches.^2,2));
patch_idx = patch_norms > 1e-4;
patches = patches(patch_idx,:);
patch_norms = sqrt(sum(patches.^2,2));
fprintf('  Computing features {\n');
if (feat_type == 1)
   % Compute covariance features
   for i=1:basis_count,
       pf = (squeeze(A(:,:,i)) * patches')';
       pf_norms = sqrt(sum(pf.^2,2));
       dots = sum(pf .* patches,2) ./ (patch_norms .* pf_norms);
       patch_feats(patch_idx,i) = dots;
       fprintf('    basis-%d\n',i);
   end
else
    patch_feats(patch_idx,:) = patches * A;
end
fprintf('  }\n');
return
end
