%%%%%%%%%%%%%%%%
% Scrap script %
%%%%%%%%%%%%%%%%

% Get sample feature vectors for both basis types, to find the dimension of the
% feature vectors for each basis type.
cov_feats = cc_patch_features( X_mnist(1,:), A_cov, W );
omp_feats = cc_patch_features( X_mnist(1,:), A_omp, W );

% Set up train/test parameters
train_size = 7500;
Xt_cov = zeros(train_size,numel(cov_feats));
Xt_omp = zeros(train_size,numel(omp_feats));
Yt = zeros(train_size,1);
idx_train = randsample(1:60000,train_size);

% Convert training/testing images to both feature representations
fprintf('Converting ims => features:');
for i=1:train_size,
    cov_feats = cc_patch_features( X_mnist(idx_train(i),:), A_cov, W );
    omp_feats = cc_patch_features( X_mnist(idx_train(i),:), A_omp, W );
    Xt_cov(i,:) = cov_feats;
    Xt_omp(i,:) = omp_feats;
    Yt(i) = Y_mnist(idx_train(i));
    if (mod(i,round(train_size / 50)) == 0)
        fprintf('.');
    end
end
fprintf('\n');
save('mnist_test.mat');

% Test simple SVM on both feature types, and also on union of feature types
Xt_cov = sparse(ZMUV(Xt_cov));
Xt_omp = sparse(ZMUV(Xt_omp));
fprintf('==================================================\n');
fprintf('Testing COV features:\n');
train(Yt,Xt_cov,'-s 0 -v 10');
fprintf('==================================================\n');
fprintf('Testing OMP features:\n');
train(Yt,Xt_omp,'-s 0 -v 10');
fprintf('==================================================\n');
fprintf('Testing JOINT features:\n');
train(Yt,[Xt_cov Xt_omp],'-s 0 -v 10');