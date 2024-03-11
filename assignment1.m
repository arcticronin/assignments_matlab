accuracy_5x2 = [];

for ndataset = 1:4
    switch ndataset
        case 1, load Datasets/dataset1.mat
        case 2, load Datasets/dataset2.mat
        case 3, load Datasets/dataset3.mat    
        case 4, load Datasets/dataset4.mat
        otherwise
    end
    accuracy_times = [];

    for ntimes = 1:5
        % stratified sampling
        idx_tr=[];
        idx_te=[];

        for nclass = 1:2
            u = find( labels == nclass);
            idx = randperm( numel (u));
            idx_tr = [idx_tr; u(idx( 1:round(numel(idx)/2)))];
            idx_te = [idx_te; u(idx( 1+round(numel(idx)/2): end))];
        end

        labels_tr = labels(idx_tr);
        labels_te = labels(idx_te);
        data_tr = data(idx_tr,:);
        data_te = data(idx_te,:);

        % training calssifier(s)
        % train on training split, test on test split
        SVM_LIN = fitcsvm(data_tr, labels_tr, 'KernelFunction',...
            'linear', 'KernelScale', 1);

        % second classifyer
        SVM_RBF = fitcsvm(data_tr, labels_tr, 'KernelFunction',...
            'gaussian', 'KernelScale', 0.1);
       
        % k nearest neighbor
        % hyperpars =  Euclidean Distance , k=10
        KNN = fitcknn(data_tr, labels_tr, 'Distance',...
            'Euclidean', 'Numneighbors', 10);
       
        % tree
        % hyperpars max number of splits = 15
        TREE = fitctree(data_tr, labels_tr, 'SplitCriterion',...
            'gdi', 'MaxNumSplits', 15);
    
        % predicting 1
        
        % accuracies = [];

        prediction = predict(SVM_LIN, data_te);
        accuracy1 = numel(find(prediction == labels_te))...
            / numel(labels_te);

        prediction = predict(SVM_LIN, data_te);
        accuracy1 = numel(find(prediction == labels_te))...
            / numel(labels_te);
        
        prediction = predict(SVM_LIN, data_te);
        accuracy1 = numel(find(prediction == labels_te))...
            / numel(labels_te);
        
        prediction = predict(SVM_LIN, data_te);
        accuracy1 = numel(find(prediction == labels_te))...
            / numel(labels_te);

        % training calssifier(s) 
        % reversing order of the splits
        SVM_LIN = fitcsvm(data_te, labels_te, 'KernelFunction',...
            'linear', 'KernelScale', 1);
        SVM_RBF = fitcsvm(data_te, labels_te, 'KernelFunction',...
            'gaussian', 'KernelScale', 0.1);
        KNN = fitcknn(data_te, labels_te, 'Distance',...
            'Euclidean', 'Numneighbors', 10);
        TREE = fitctree(data_te, labels_te, 'SplitCriterion',...
            'gdi', 'MaxNumSplits', 15);

        % predicting 2

        prediction = predict(SVM_LIN, data_tr);
        accuracy2 = numel(find(prediction == labels_tr))...
            / numel(labels_tr);

        % averaging
        
        accuracy = (accuracy1 + accuracy2) / 2;
        accuracy_times(ntimes, 1) = accuracy;


    end

    accuracy_5x2(ndataset, 1) = mean(accuracy_times);

end