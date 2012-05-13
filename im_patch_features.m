function [ im_feats ] = im_patch_features( im, A, split_neg, grid_count, W )
% Generate features for a B/W image, given a (1 x d*d) uint8 representation
% of the image. Bases for feature generation are given in A, from which we can
% infer the patch size. Use the whitening transformation given in W on each
% patch prior to feature computation.
%
% Parameters:
%   im: the image for which to compute patch features (1 x im_dim^2)
%   A: the bases to use for covcode features (w^2 x w^2 x basis_count)
%   split_neg: 0/1, determines whether to split each features response into
%              separate positive and negative components
%   grid_count: the grid count along each dimension for feature aggregation
%   W: an optional whitening transform for preprocessing (w^2 x w^2)
% Outputs:
%   im_feats: covcode features for the image
%

im_dim = round(sqrt(numel(im)));
if (abs(im_dim - sqrt(numel(im))) > 0.01)
    error('patch_features: function only valid for square images.\n');
end
% Reshape image to native square/2d form
im = double(reshape(im, im_dim, im_dim));
% Get the number of bases and the size of patches to process
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
        error('patch_features: A should be either 2 or 3 dimensional.\n');
    end
end
w = round(sqrt(size(A,1)));
ws = w - 1;
patch_pix = w*w;
% Check if a whitening transform was given for preprocessing
if ~exist('W','var')
    W = eye(patch_pix);
else
    if (size(W,1) ~= patch_pix || size(W,2) ~= patch_pix)
        error('patch_features: whitener has the wrong dimension\n');
    end
end

min_coord = 1;
max_coord = im_dim - ws;
start_coords = min_coord:max_coord;
% Create a map for identifying the zone to which each patch belongs, for now
% there will be four zones (upper-left, upper-right, lower-left, lower-right).
zone_map = ones(numel(start_coords),numel(start_coords));
nsc = numel(start_coords);
z_coords = cell(grid_count,1);
for z=1:grid_count,
    z_start = round(((z-1) * nsc) / grid_count) + 1;
    if (z < grid_count)
        z_end = round((z * nsc) / grid_count);
    else
        z_end = nsc;
    end
    z_coords{z} = z_start:z_end;
end
z_idx = 1;
for z1=1:grid_count,
    for z2=1:grid_count,
        z1_coords = z_coords{z1};
        z2_coords = z_coords{z2};
        zone_map(z1_coords,z2_coords) = z_idx;
        z_idx = z_idx + 1;
    end
end
zone_count = numel(unique(zone_map(:)));

patch_vals = zeros(numel(start_coords)*numel(start_coords),patch_pix);
patch_zones = zeros(numel(start_coords)*numel(start_coords),1);
patch_idx = 1;
for row_i=1:numel(start_coords),
    row = start_coords(row_i);
    for col_i=1:numel(start_coords),
        col = start_coords(col_i);
        % Extract the patch at this location
        patch = reshape(im(row:(row+ws),col:(col+ws)),1,patch_pix);
        patch = patch - (sum(patch) / patch_pix);
        pnorm = norm(patch);
        if (pnorm > 1e-3)
            patch = patch ./ pnorm;
            % Apply whitening transform (or identity transform if no W given)
            patch = patch * W;
        end
        patch_vals(patch_idx,:) = patch(:);
        patch_zones(patch_idx) = zone_map(row,col);
        patch_idx = patch_idx + 1;
    end
end
patch_feats = compute_features(patch_vals,A,basis_count,feat_type,split_neg);
im_feats = zeros(zone_count,size(patch_feats,2));
for i=1:zone_count,
    im_feats(i,:) = zonal_features(patch_feats(patch_zones == i,:));
end
im_feats = im_feats(:);

return

end

function [ patch_feats ] = compute_features(...
    patches, A, basis_count, feat_type, split_neg)
% Compute features for the given patches, given the bases in A.
%
% Parameters:
%   patches: patches for which to compute features (patch_count x patch_pix)
%   A: bases for features (patch_pix (x patch_pix) x basis_count)
%   basis_count: the number of bases in A
%   feat_type: whether to compute linear or covariance features (1 or 2)
%   split_neg: 0/1, determines whether to split each features response into
%              separate positive and negative components
% Outputs:
%   patch_feats: the features computed per-patch (patch_count x basis_count)
%
patch_feats = zeros(size(patches,1),basis_count);
patch_norms = sqrt(sum(patches.^2,2));
patch_idx = patch_norms > 1e-3;
patches = patches(patch_idx,:);
patch_norms = sqrt(sum(patches.^2,2));
if (feat_type == 1)
   % Compute covariance features
   for i=1:basis_count,
       pf = (squeeze(A(:,:,i)) * patches')';
       pf_norms = sqrt(sum(pf.^2,2));
       dots = sum(pf .* patches,2) ./ (patch_norms .* pf_norms);
       patch_feats(patch_idx,i) = dots;
   end
else
    patch_feats(patch_idx,:) = patches * A;
end
if (split_neg == 1)
    % Split patch features into negative and positive components
    pf1 = patch_feats;
    pf2 = -patch_feats;
    pf1(pf1 < 0) = 0;
    pf2(pf2 < 0) = 0;
    patch_feats = [pf1 pf2];
end
return
end

function [ zone_feats ] = zonal_features( patch_feats )
% Compute a set of "zonal aggregate" features given the features for each patch
% in the given zone. For now, compute soft-max-like features.
%
% Parameters;
%   patch_feats: base features for each patch in the zone
% Outputs:
%   zone_feats: aggregate features for the zone
%
alpha = 1.0;
feat_exp = exp(abs(patch_feats .* alpha));
zone_feats = sum(patch_feats .* feat_exp) ./ sum(feat_exp);
return
end


