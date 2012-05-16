% Do a simple test of covariance feature learning using MNIST digit data
clear;
load('cifar_color_data.mat');

%%%%%%%%%%%%%%%%%%
% Basis Learning %
%%%%%%%%%%%%%%%%%%

% Set up parameters for covariance code basis learning
patch_count = 7500;
patch_size = 5;
round_count = 100;
basis_count = 100;
grid_count = 2;
cov_spars = 5; %(16 / basis_count);
lam_l1 = 1e-4;
step = 20;
split_neg = 1;
omp_count = 5;
omp_patches = 10000;
omp_step = 5;
omp_l1 = 1e-4;
thresh_rates = [0.25 0.30 0.35 0.40 0.45];

% Compute a whitening transform to use with patches
[ patches ] = extract_patches(Xtr_cifar, patch_size, 250000, 32, 32, 3);
[ W ] = compute_whitener( patches );

% Learn covariance coding features using whitened sampled patches
for r=1:round_count,
    fprintf('================\n');
    fprintf('OUTER ROUND %d |\n',r);
    fprintf('================\n');
    patches = extract_patches(Xtr_cifar, patch_size, patch_count, 32, 32, 3);
    patches = bsxfun(@minus,patches,W.M) * W.W';
    if (r == 1)
        [ A_cov ] = learn_cov_bases(...
            patches, basis_count, cov_spars, lam_l1, step, 1);
    else
        [ A_cov ] = learn_cov_bases(...
            patches, basis_count, cov_spars, lam_l1, step, 1, A_cov);
    end
end

% Learn linear reconstruction bases via Orthogonal Matching Pursuit
for r=1:round(round_count),
    fprintf('================\n');
    fprintf('OUTER ROUND %d |\n',r);
    fprintf('================\n');
    patches = extract_patches(Xtr_cifar, patch_size, omp_patches, 32, 32, 3);
    patches = bsxfun(@minus,patches,W.M) * W.W';
    if (r == 1)
        [ A_omp ] = learn_omp_bases(...
            patches, basis_count, omp_count, omp_step, 1, omp_l1 );
    else
        [ A_omp ] = learn_omp_bases(...
            patches, basis_count, omp_count, omp_step, 1, omp_l1, A_omp );
    end
end

% Save learned feature matrices and such
save('cifar_test.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Extraction and Classification %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute triangle activation thresholds based on a sampled set of patches
[ patches ] = extract_patches(Xtr_cifar, patch_size, 25000, 32, 32, 3);
[ threshs_cov ] = compute_thresholds( patches, A_cov, thresh_rates, W );
[ threshs_omp ] = compute_thresholds( patches, A_omp, thresh_rates, W );

% Get sample feature vectors for both basis types, to find the dimension of the
% feature vectors for each basis type.
cov_feats = im_patch_features_color(...
    Xtr_cifar(1,:), A_cov, threshs_cov(3), split_neg, grid_count, W);
omp_feats = im_patch_features_color(...
    Xtr_cifar(1,:), A_omp, threshs_omp(3), split_neg, grid_count, W);

% Set up train/test parameters
train_size = 10000;
idx_train = randsample(1:50000,train_size);
Xtr_cifar = Xtr_cifar(idx_train,:);
Ytr_cifar = Ytr_cifar(idx_train);
Xt_cov = zeros(train_size,numel(cov_feats));
Xt_omp = zeros(train_size,numel(omp_feats));
Yt = zeros(train_size,1);

% Convert training/testing images to both feature representations
fprintf('Converting ims => features:');
for i=1:train_size,
    %cov_feats = im_patch_features_color(...
    %    Xtr_cifar(i,:), A_cov, threshs_cov(3), split_neg, grid_count, W);
    omp_feats = im_patch_features_color(...
        Xtr_cifar(i,:), A_omp, threshs_omp(3), split_neg, grid_count, W);
    %Xt_cov(i,:) = cov_feats;
    Xt_omp(i,:) = omp_feats;
    Yt(i) = Ytr_cifar(i);
    if (mod(i,round(train_size / 50)) == 0)
        fprintf('.');
    end
end
fprintf('\n');

% Save computed image features
save('cifar_test.mat');

% Test simple SVM on both feature types, and also on union of feature types
Xt_cov = sparse(ZMUV(Xt_cov));
Xt_omp = sparse(ZMUV(Xt_omp));
fprintf('==================================================\n');
fprintf('Testing COV features:\n');
train(Yt,Xt_cov,'-s 4 -v 5');
fprintf('==================================================\n');
fprintf('Testing OMP features:\n');
train(Yt,Xt_omp,'-s 4 -v 5');
% fprintf('==================================================\n');
% fprintf('Testing JOINT features:\n');
% train(Yt,[Xt_cov Xt_omp],'-s 4 -v 5');




%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
