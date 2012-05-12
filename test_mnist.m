% Do a simple test of covariance feature learning using MNIST digit data
clear;
load('mnist_data.mat');

%%%%%%%%%%%%%%%%%%
% Basis Learning %
%%%%%%%%%%%%%%%%%%

% Set up parameters for covariance code basis learning
patch_count = 5000;
patch_size = 8;
round_count = 300;
basis_count = 256;
spars = 16 / basis_count;
lam_l1 = 1e-5;
step = 16;
omp_count = 5;
omp_patches = 10000;
omp_step = 1;

% Compute a whitening transform to use with patches
[ patches ] = extract_patches(X_mnist, patch_size, 250000, 28, 28, 1);
[ W ] = compute_whitener( patches );

% Learn covariance coding features using whitened sampled patches
for r=1:round_count,
    fprintf('================\n');
    fprintf('OUTER ROUND %d |\n',r);
    fprintf('================\n');
    [ patches ] = extract_patches(X_mnist, patch_size, patch_count, 28, 28, 1);
    patches = patches * W';
    if (r == 1)
        [ A_cov ] = learn_cov_bases(...
            patches, basis_count, spars, lam_l1, step, 1);
    else
        [ A_cov ] = learn_cov_bases(...
            patches, basis_count, spars, lam_l1, step, 1, A_cov);
    end
end

% Learn linear reconstruction bases via Orthogonal Matching Pursuit
for r=1:round(round_count),
    fprintf('================\n');
    fprintf('OUTER ROUND %d |\n',r);
    fprintf('================\n');
    [ patches ] = extract_patches(X_mnist, patch_size, omp_patches, 28, 28, 1);
    patches = patches * W';
    if (r == 1)
        [ A_omp ] = learn_omp_bases(...
            patches, basis_count, omp_count, omp_step, 1 );
    else
        [ A_omp ] = learn_omp_bases(...
            patches, basis_count, omp_count, omp_step, 1, A_omp );
    end
end

% Save learned feature matrices and such
save('mnist_test.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Extraction and Classification %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sample feature vectors for both basis types, to find the dimension of the
% feature vectors for each basis type.
cov_feats = im_patch_features( X_mnist(1,:), A_cov, W );
omp_feats = im_patch_features( X_mnist(1,:), A_omp, W );

% Set up train/test parameters
train_size = 10000;
Xt_cov = zeros(train_size,numel(cov_feats));
Xt_omp = zeros(train_size,numel(omp_feats));
Yt = zeros(train_size,1);
idx_train = randsample(1:60000,train_size);

% Convert training/testing images to both feature representations
fprintf('Converting ims => features:');
for i=1:train_size,
    cov_feats = im_patch_features( X_mnist(idx_train(i),:), A_cov, W );
    omp_feats = im_patch_features( X_mnist(idx_train(i),:), A_omp, W );
    Xt_cov(i,:) = cov_feats;
    Xt_omp(i,:) = omp_feats;
    Yt(i) = Y_mnist(idx_train(i));
    if (mod(i,round(train_size / 50)) == 0)
        fprintf('.');
    end
end
fprintf('\n');

% Save computed image features
save('mnist_test.mat');

% Test simple SVM on both feature types, and also on union of feature types
Xt_cov = sparse(ZMUV(Xt_cov));
Xt_omp = sparse(ZMUV(Xt_omp));
fprintf('==================================================\n');
fprintf('Testing COV features:\n');
train(Yt,Xt_cov,'-s 2 -v 5');
fprintf('==================================================\n');
fprintf('Testing OMP features:\n');
train(Yt,Xt_omp,'-s 2 -v 5');
fprintf('==================================================\n');
fprintf('Testing JOINT features:\n');
train(Yt,[Xt_cov Xt_omp],'-s 2 -v 5');




%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%