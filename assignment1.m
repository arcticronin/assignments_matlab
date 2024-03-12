
accuracy_5x2 = [];

for ndataset = 1:4 % todo swap back to 4
    % loads in variables data, labels
    switch ndataset
        case 1, load Datasets/dataset1.mat
        case 2, load Datasets/dataset2.mat
        case 3, load Datasets/dataset3.mat    
        case 4, load Datasets/dataset4.mat
        otherwise
    end

    accuracy_times = [];
    accuracy_arrays = [];

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

        models = fitmodels(data_tr, labels_tr);
        % just for compatibility, todo remove
        SVM_LIN = fitcsvm(data_tr, labels_tr, 'KernelFunction',...
            'linear', 'KernelScale', 1);

        predictionsTable = table();   
        accuraciesTable = table();

        for m = 1:length(models)
            % Obtain predictions for the current model
            % currentPredictions = predict(models{m}.model, data_te);
            % for debugging reasons
            % currentPredictionsTable = array2table(currentPredictions, ...
            %                                      'VariableNames', ...
            %                                      models{m}.name);
            % predictionsTable = [predictionsTable, currentPredictionsTable];
           
            accuracy = sum( predict(models{m}.model, data_te)== ...
                labels_te)/ numel(labels_te);
            
            % Convert the accuracy into a table
            currentAccuracyTable = array2table(accuracy, ...
                'VariableNames', models{m}.name);
            
            % Concatenate the current model's accuracy to the accuracies table
            % Since accuraciesTable has a single row, now
            % we can concatenate horiz
            accuraciesTable = [accuraciesTable, currentAccuracyTable];
            
        end

        prediction = predict(SVM_LIN, data_te);
        accuracy1 = numel(find(prediction == labels_te))...
             / numel(labels_te);


        % training calssifier(s) 
        models = fitmodels(data_te, labels_te);

        temp_acc = zeros(1,length(models));
        for m = 1:length(models)
            % Obtain predictions for the current model
            % currentPredictions = predict(models{m}.model, data_te);
            % for debugging reasons
            % currentPredictionsTable = array2table(currentPredictions, ...
            %                                      'VariableNames', ...
            %                                      models{m}.name);
            % predictionsTable = [predictionsTable, currentPredictionsTable];
           
            accuracy = sum( predict(models{m}.model, data_te)== ...
                labels_te)/ numel(labels_te);
           
            % Concatenate the current model's accuracy to the accuracies table
            % Since accuraciesTable has a single row, we can horizontally concatenate
            temp_acc(1,m) = accuracy;
        end
        accuraciesTable{2,:} = temp_acc; % do it in one shot helps fro warnings

        % averageAccuracies = varfun(@mean, accuraciesTable);
        
        averageAccuraciesArray = mean(table2array(accuraciesTable));
        accuracy_arrays(ntimes, :) = averageAccuraciesArray;
        
        % predicting 2
        prediction = predict(SVM_LIN, data_tr);
        accuracy2 = numel(find(prediction == labels_tr))...
            / numel(labels_tr);

        % averaging 
        accuracy = (accuracy1 + accuracy2) / 2;
        accuracy_times(ntimes, 1) = accuracy;

    end

    accuracy_5x2(ndataset, 1) = mean(accuracy_times);
    accuracy_5x2_all(ndataset,:) = mean(accuracy_arrays);

end

finalresult = arr2modeldf(accuracy_5x2_all, models);
rankedmatrix = rankRowsAscending(accuracy_5x2_all); %% todo doesnt work


function s = fitmodels(X, y)

        % training calssifier(s)
        SVM_LIN = fitcsvm(X, y, 'KernelFunction',...
            'linear', 'KernelScale', 1);

        SVM_RBF = fitcsvm(X, y, 'KernelFunction',...
            'gaussian', 'KernelScale', 0.1);

        % k nearest neighbor
        % hyperpars =  Euclidean Distance , k=10
        KNN = fitcknn(X, y, 'Distance',...
            'Euclidean', 'Numneighbors', 10);

        % tree
        % hyperpars max number of splits = 15
        TREE = fitctree(X, y, 'SplitCriterion',...
            'gdi', 'MaxNumSplits', 15);
        
        m1.name = "svm_lin";
        m1.model = SVM_LIN;
        m2.name = "svm_rbf";
        m2.model = SVM_RBF;
        m3.name = "knn";
        m3.model = KNN;
        m4.name = "tree";
        m4.model = TREE;

        s = {m1, m2, m3, m4};
        % later access to them by s{1}.name
end

function df = arr2modeldf(m, models)
    n = length(models);
    disp(models{1}.name);
    
    names = {};
    
    for i = 1:n
        names{i} = models{i}.name{1};
        disp(names);
    end
 
    df = array2table(m, 'VariableNames', names);
end

function rankMatrix = rankRowsAscending(matrix)
    % Get the size of the matrix
    [nRows, nCols] = size(matrix);
    
    % Preallocate the rank matrix
    rankMatrix = zeros(nRows, nCols);
    
    % Loop through each row
    for i = 1:nRows
        % Sort the row to get the original indices of the sorted elements
        [~, sortedIndices] = sort(matrix(i, :));
        
        % Now, assign ranks based on the sorted indices
        % Since MATLAB indices start at 1, we subtract 1 to start ranks at 0
        ranks = sortedIndices - 1;
        
        % To place ranks in their original position, we sort the ranks based on the sorted indices
        rankMatrix(i, sortedIndices) = ranks;
    end
end
