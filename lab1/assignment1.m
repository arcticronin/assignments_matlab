
for ndataset = 1:4 % todo swap back to 4
    % loads in variables data, labels
    switch ndataset
        case 1, load Datasets/dataset1.mat
        case 2, load Datasets/dataset2.mat
        case 3, load Datasets/dataset3.mat    
        case 4, load Datasets/dataset4.mat
        otherwise
    end

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
        
        if ndataset==1
            % plotgrid(models{2}.model, data, labels);
        end
        predictionsTable = table();   
        accuraciesTable = table();

        for m = 1:length(models)
            % Obtain predictions for the current model
           
            accuracy = sum( predict(models{m}.model, data_te)== ...
                labels_te)/ numel(labels_te);
            
            % Convert the accuracy into a table
            currentAccuracyTable = array2table(accuracy, ...
                'VariableNames', models{m}.name);
            
           accuraciesTable = [accuraciesTable, currentAccuracyTable];
            
        end

                % training calssifier(s) 
        models = fitmodels(data_te, labels_te);

        temp_acc = zeros(1,length(models));
        for m = 1:length(models)
           
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
        
    end
    
    accuracy_5x2_all(ndataset,:) = mean(accuracy_arrays);

end




ranked_matrix = rank_matrix(accuracy_5x2_all);

ranked_df = arr2modeldf(ranked_matrix, models);

finalresult = arr2modeldf(accuracy_5x2_all, models);

cv = compute_CV(accuracy_5x2_all);

% plottadataset();

plot_err_bars(rank_matrix(accuracy_5x2_all), cv);

function s = fitmodels(X, y)

        % training calssifier(s)
        SVM_LIN = fitcsvm(X, y, 'KernelFunction',...
            'linear', 'KernelScale', 2);

        SVM_RBF = fitcsvm(X, y, 'KernelFunction',...
            'gaussian', 'KernelScale', 0.82);

        % k nearest neighbor
        % hyperpars =  Euclidean Distance , k=10
        KNN = fitcknn(X, y, 'Distance',...
            'Euclidean', 'Numneighbors', 12);

        % tree
        % hyperpars max number of splits = 15
        TREE = fitctree(X, y, 'SplitCriterion',...
            'gdi', 'MaxNumSplits', 11);
        
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

function ranked = rank_row(a)
    d = dictionary('KeyType', 'double', 'ValueType', 'double');
    sa = sort(a, 'descend');
    for i = 1:length(a)
        if ~ d.isKey(a(i))
            d(a(i)) = mean(find(sa == a(i)));       
        end
    end
    ranked = zeros(1,length(a));
    for i = 1:length(a)
       ranked(1,i) = d(a(i));
    end
    disp(d);
end

function ranked = rank_matrix(m)
    [nRows, nCols] = size(m);
    ranked = zeros(nRows, nCols);
    for i = 1:nRows
        ranked(i, :) = rank_row(m(i, :));
    end
end

function cv = compute_CV(m)
    [N, k] = size(m);
    q = 2.569; % q alpha f
    %q = 2.291; % q
    cv = q * sqrt((k * (k+1))/(6 * N));
end

function plot_err_bars(m, cv)
    errorbar(mean(m), 1:4, cv/2, 'horizontal', 'Marker', 'square', ...
             'LineWidth', 1.5, 'MarkerSize', 10);
    xlabel("Rank"); % Label for the x-axis
    % Uncomment the following line to set the y-axis label
    % ylabel("Classifier");
    ylim([0.1, 4.9]); % Set y-axis limits
    xlim([0.6, 4.4]); % Set x-axis limits
    labels = {"svm\_lin", "svm\_rbf", "knn", "tree"}; % Define labels for each classifier
    yticks(1:4); % Set ticks on the y-axis at positions 1 through 4
    yticklabels(labels); % Apply the labels to these ticks
end


function plottadataset()
        load Datasets/dataset1.mat;
        subplot(2,2,1);
        gscatter(data(:,1), data(:,2), labels);
        title("dataset\_1");
        load Datasets/dataset2.mat;
        subplot(2,2,2);
        gscatter(data(:,1), data(:,2), labels);
        title("dataset\_2");
        load Datasets/dataset3.mat  ;
        subplot(2,2,3);
        gscatter(data(:,1), data(:,2), labels);
        title("dataset\_3");
        load Datasets/dataset4.mat;
        subplot(2,2,4);
        gscatter(data(:,1), data(:,2), labels) ; 
        title("dataset\_4");
end

% from matlab 2 class classifier documentation
function plotgrid(cl, data, labels)
    % Predict scores over the grid
    d = 0.02;
    [x1Grid,x2Grid] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
        min(data(:,2)):d:max(data(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    [~,scores] = predict(cl,xGrid);
    
    % Plot the data and the decision boundary
    figure;
    h(1:2) = gscatter(data(:,1),data(:,2),labels, "rb",'.');
    hold on
    h(3) = plot(data(cl.IsSupportVector,1),data(cl.IsSupportVector,2),'ko');
    contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
    legend(h,{'class 1','class 2','Support Vectors'});
    axis equal
    hold off
end


%% gridearch

% gridsearch = dictionary('KeyType', 'double', 'ValueType', 'double');
function gridsearch = gridsearchforhyperpar(rangeh)
    gridsearch = dictionary();
    gridsearch(0.001) = 0.001;
    
    for hyperpar = rangeh %0.02:0.02:2
        for ndataset = 1:4 % todo swap back to 4
            % loads in variables data, labels
            switch ndataset
                case 1, load Datasets/dataset1.mat
                case 2, load Datasets/dataset2.mat
                case 3, load Datasets/dataset3.mat    
                case 4, load Datasets/dataset4.mat
                otherwise
            end
        
            accuracy_arrays = [];
        
            for ntimes = 1:20
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
        
                models = fitmodel(data_tr, labels_tr, hyperpar);
               
        
                predictionsTable = table();   
                accuraciesTable = table();
        
                for m = 1:length(models)
                    % Obtain predictions for the current model
                   
                    accuracy = sum( predict(models{m}.model, data_te)== ...
                        labels_te)/ numel(labels_te);
                    
                    % Convert the accuracy into a table
                    currentAccuracyTable = array2table(accuracy, ...
                        'VariableNames', models{m}.name);
                    
                   accuraciesTable = [accuraciesTable, currentAccuracyTable];
                    
                end
        
                        % training calssifier(s) 
                models = fitmodel(data_te, labels_te, hyperpar);
        
                temp_acc = zeros(1,length(models));
                for m = 1:length(models)
                   
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
                
            end
        
            accuracy_5x2_all(ndataset,:) = mean(accuracy_arrays);
        
        end
        if ~ gridsearch.isKey(mean(accuracy_arrays))
            gridsearch(mean(accuracy_arrays)) = hyperpar;
        end
    end
end

%max_acc = max(gridsearch.keys);
%best_hyperpar = gridsearch(max_acc);
function s = fitmodel(X, y, hyperpar)
        % training calssifier(s)
        SVM_LIN = fitcsvm(X, y, 'KernelFunction',...
            'linear', 'KernelScale', 1); %1
        SVM_RBF = fitcsvm(X, y, 'KernelFunction',...
            'gaussian', 'KernelScale', hyperpar); %0.1
        % k nearest neighbor
        % hyperpars =  Euclidean Distance , k=10
        KNN = fitcknn(X, y, 'Distance',...
            'Euclidean', 'Numneighbors', 10);
        % tree
        % hyperpars max number of splits = 15
        TREE = fitctree(X, y, 'SplitCriterion',...
            'gdi', 'MaxNumSplits', 25);
        m1.name = "svm_lin";
        m1.model = SVM_LIN;
        m2.name = "svm_rbf";
        m2.model = SVM_RBF;
        m3.name = "knn";
        m3.model = KNN;
        m4.name = "tree";
        m4.model = TREE;
        stot = {m1, m2, m3, m4};
        t = 1;
        s = {stot{1}};
end

