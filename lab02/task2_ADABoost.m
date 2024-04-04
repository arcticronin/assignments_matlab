%% Homework
% TBD compare the performance of the meta-classifier when trained on
% Predictions (i.e the predicted class) instead the classification Scores

% TBD compare the performance of the meta classifier when the training
% split is not performad and the same data is used to train the level-1
% classifiers and the meta classifier

% results = dictionary();
%% Stack Classifiers

load dataset.mat

% already has data_tr and data_te, 
% labels_tr and labels_te
% 1200 train and 1200 test
%%

% u=find(labels_tr==1);
% figure(1),
% hold on
% plot(data_tr(u,1), data_tr(u,2), 'r.')
% u=find(labels_tr==2);
% plot(data_tr(u,1), data_tr(u,2), 'b.')
% hold off

%%
% right procedure: train classifiers on different data (on 2 different folds)
% train cl 1 on first fold, make the predictions on second fold,
% train cl 2 on the prediction obtained in step 1

% random number generator 
rng('default'); %% for reproducibility

idx_f1 = [];
idx_f2 = [];

for nclass = 1:2
    u = find(labels_tr == nclass);
    idx = randperm(numel(u));
    % so I know the data is balanced, stratifyed sampling
    idx_f1 = [idx_f1; u(idx(1:round(numel(idx)/2)))];
    idx_f2 = [idx_f2; u(idx(1+round(numel(idx)/2):end))];
end

% training data
labels_f1 = labels_tr(idx_f1);
labels_f2 = labels_tr(idx_f2);

data_f1 = data_tr(idx_f1,:);
data_f2 = data_tr(idx_f2,:);

%% train level 1 classifiers on fold 1 
mdl = {};

% SVM with gaussian kernel
rng('default');
mdl{1} = fitcsvm(data_f1, labels_f1, 'KernelFunction','gaussian', ...
    'KernelScale', 5);

% SVM with gaussian kernel
rng('default');
mdl{2} = fitcsvm(data_f1, labels_f1, 'KernelFunction','polynomial', ...
    'KernelScale', 10);

% decision tree
rng('default');
mdl{3} = fitctree(data_f1, labels_f1, 'SplitCriterion', ...
    'gdi', 'MaxNumSplits', 20);

% Naive Bayes
rng('default');
mdl{4} = fitcnb(data_f1, labels_f1);

% ensemble of decision tree
rng('default');
mdl{5} = fitcensemble( data_f1, labels_f1);

%% predictions on fold 2 (to be used to train the models) 

N = numel(mdl);

Predictions = zeros(size(data_f2,1),N);
% scores are one step before the predictions, but MORE INFORMATIVE in{-inf, +inf}
Scores = zeros(size(data_f2, 1), N);

for ii = 1:N
    [predictions, scores] = predict(mdl{ii}, data_f2);
    Predictions(:,ii) = predictions;
    Scores(:,ii) = scores(:,1);
end
% i obtained the features where to train second level classifier

%% train the stacked classifier on fold 2
rng("default");
% stackedModel = fitcensemble(Predictions, labels_f2, "Method", "Bag");
stackedModel = fitcensemble(Scores, labels_f2, "Method", "AdaBoostM1");

mdl{N+1} = stackedModel;

% array for accuracy
ACC = [];
Predictions = zeros(size(data_te,1),N);
% scores are one step before the predictions
Scores = zeros(size(data_te, 1), N);

for ii = 1:N
    [predictions, scores] = predict(mdl{ii}, data_te);
    Predictions(:,ii) = predictions;
    Scores(:,ii) = scores(:,1);
    ACC(ii) = numel(find(predictions == labels_te))...
                / numel(labels_te);
end

predictions = predict(mdl{N+1}, Predictions);
ACC(N+1) = numel(find(predictions == labels_te)) ...
                / numel(labels_te);


% results{"correct approach"} = ACC;
results{"trained on prediction (less informative) + ADABoost"} = ACC;
% results{"no folds"} = ACC;
