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

%% train level 1 classifiers on fold 1 
mdl = {};

% SVM with gaussian kernel
rng('default');
mdl{1} = fitcsvm(data_tr, labels_tr, 'KernelFunction','gaussian', ...
    'KernelScale', 5);

% SVM with gaussian kernel
rng('default');
mdl{2} = fitcsvm(data_tr, labels_tr, 'KernelFunction','polynomial', ...
    'KernelScale', 10);

% decision tree
rng('default');
mdl{3} = fitctree(data_tr, labels_tr, 'SplitCriterion', ...
    'gdi', 'MaxNumSplits', 20);

% Naive Bayes
rng('default');
mdl{4} = fitcnb(data_tr, labels_tr);

% ensemble of decision tree
rng('default');
mdl{5} = fitcensemble( data_tr, labels_tr);

%% predictions on the same fold (to be used to train the models) 

N = numel(mdl);

Predictions = zeros(size(data_tr,1),N);
% scores are one step before the predictions, but MORE INFORMATIVE in{-inf, +inf}
Scores = zeros(size(data_tr, 1), N);

for ii = 1:N
    [predictions, scores] = predict(mdl{ii}, data_tr);
    Predictions(:,ii) = predictions;
    Scores(:,ii) = scores(:,1);
end
% i obtained the features where to train second level classifier





%% train the stacked classifier on fold 2





rng("default");
stackedModel = fitcensemble(Predictions, labels_te, "Method", "Bag");
% stackedModel = fitcensemble(Scores, labels_te, "Method", "Bag");
% stackedModel = fitcensemble(Scores, labels_te, "Method", "AdaBoostM1");

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
% results{"trained on prediction (less informative)"} = ACC;
results{"no folds, stacked trained on prediction"} = ACC;
