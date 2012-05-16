function [ opt_theta acc_tr acc_te svm_eval ] = svm_cross_validate(...
    X, Y, cv_rounds, cv_frac, lams )
% Cross-validate simple 1-vs-all L2-SVM using the observations/classes in X/Y,
% performing cross-validation for each regularization weight in lams
%
% Parameters:
%   X: set of (row-wise) training observations
%   Y: set of training classes
%   cv_rounds: number of cross-validation rounds to peform for each lambda
%   cv_frac: fraction of data to use for testing in each cv round
%   lams: sequence of lambdas for which to perform cross-validation
% Outputs:
%   opt_theta: theta learned on full observation set, using the lambda that
%              performed best in cross-validation
%   accs: test accuracy for each lambda/cv round (cv_rounds x numel(lams))
%   svm_eval: function to evaluate SVM using a theta learned using all training
%             observations and the lambda that was best in cv
%
obs_count = size(X,1);
lam_count = numel(lams);
acc_tr = zeros(cv_rounds, lam_count);
acc_te = zeros(cv_rounds, lam_count);
for r=1:cv_rounds,
    fprintf('==============\n');
    fprintf('CV ROUND %d |\n', r);
    fprintf('==============\n');
    % Sample a set of training/testing indices
    tr_idx = randsample(obs_count, round((1 - cv_frac) * obs_count));
    te_idx = setdiff(1:obs_count, tr_idx);
    % Using the sampled train/test split, do SVM for all lams
    if (r == 1)
        [opt_theta accs svm_eval] = svm_train_test(...
            X(tr_idx,:), Y(tr_idx), X(te_idx,:), Y(te_idx), lams );
    else
        [opt_theta accs svm_eval] = svm_train_test(...
            X(tr_idx,:), Y(tr_idx), X(te_idx,:), Y(te_idx), lams, opt_theta );
    end
    % Records train/test accuracies for this cross-validation round
    acc_tr(r,:) = accs(:,1)';
    acc_te(r,:) = accs(:,2)';
end
% Find the lambda producing minimum average test error
mean_accs = mean(acc_te);
[max_acc max_idx] = max(mean_accs);
% Learn an SVM using the best lambda according to CV error
[opt_theta accs svm_eval] = svm_train_test( X, Y, X, Y, lams(max_idx) );
return
end

