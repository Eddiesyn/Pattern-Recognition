% pr_exercise12_solution.m

function [res] = pr_exercise_12_solution()
    % parameters definition
    % k in kNN
    num_nn = 1;
    
    % dataset rgeneration
    % number of samples 
    samples_num = 10000;
    samples = [];
    labels = [];
    rng(0); % set random seed for reproduceability
    for i=1:samples_num
        % Sample() now returns (x, y) coordinates as well the class label
        [samples(i, :), labels(i)] = Sample();
    end
    
    % Split data into test and train sets
    train_share = 0.5;
    train_indices = 1:(train_share * samples_num);
    test_indices = (train_share * samples_num + 1):samples_num;
    
    train_X = samples(train_indices, :);
    train_labels = labels(train_indices);
    test_X = samples(test_indices, :);
    test_labels = labels(test_indices);
    
    % Visualize training data
    figure
    hold on
    plot(train_X(train_labels == 0, 1), train_X(train_labels == 0, 2), 'r*')
    plot(train_X(train_labels == 1, 1), train_X(train_labels == 1, 2), 'b*')
    
    % Get codebooks for vectors belonging to class 0 and 1 respectively
    cbook0 = clustering(train_X(train_labels == 0, :));
    cbook1 = clustering(train_X(train_labels == 1, :));
    
    % Visualize codebook vectors as bolder points (overlayed with train data points)
    plot(cbook0(:, 1), cbook0(:, 2), 'gx', 'LineWidth', 5);
    plot(cbook1(:, 1), cbook1(:, 2), 'mx', 'LineWidth', 5);
    
    disp(sprintf('%i codebook vectors for class 0 found\n%i codebook vectors for class 1 found\n', ...
        size(cbook0, 1), size(cbook1, 1)));
    
    % Run knn with all traininig set as reference vectors for respective classes
    tic % Start timer
    
    predictions = knn_fit_predict(num_nn, train_X, train_labels, test_X);
    % Calculate accuracy (an excusable metric for a toy example)
    accuracy = sum(predictions == test_labels) / size(test_labels, 2);
    disp(sprintf('Accuracy using regular kNN model: %f', accuracy));
    
    toc % Stop timer
    
    % Run knn with codebook vectors as reference vectors for respective classes
    reduced_train_X = [cbook0; cbook1];
    reduced_train_labels = [zeros(1, size(cbook0, 1)), ones(1, size(cbook1, 1))];
    tic % Start timer
    
    predictions_reduced = knn_fit_predict(num_nn, ...
                                          reduced_train_X, ...
                                          reduced_train_labels, ...
                                          test_X);
    accuracy_reduced = sum(predictions_reduced == test_labels) / size(test_labels, 2);
    disp(sprintf('Accuracy using reduced kNN model: %f', accuracy_reduced));
    
    toc % Stop timer
end

function [labels] = knn_fit_predict(k, train_X, train_labels, test_X)
    disp(sprintf('Running kNN with %i training samples\n', size(train_X, 1)));
    labels = [];
    % Iterate through all test data points
    for test_i = 1:size(test_X, 1)
        distances = [];
        % Iterater through all train samples
        for train_i = 1:size(train_X, 1)
            % Calculate the Euclidean distance between test_X(test_i, :)
            % and train_X(train_i)
            
            % distances(train_i) =  ...;
            
            distances(train_i) = sum((train_X(train_i, :) - test_X(test_i, :)).^2);
            distances(train_i) = sqrt(distances(train_i));
        end
        
        % Bind distances with labels (so that we can select the labels of the closest neighbors)
        labelled_distances = [distances', train_labels'];
        % Sort by increasing distance (1st column)
        labelled_distances = sortrows(labelled_distances, 1);
        % Take first @k labels (2nd column)
        knn_labels = labelled_distances(1:k, 2);
        
        % Binary classification with classes 0 and 1 abased on selected
        % labels
        label_count_1 = sum(knn_labels == 1);
        label_count_0 = sum(knn_labels == 0);
        
        if label_count_1 >= label_count_0
            labels(test_i) = 1;
        else
            labels(test_i) = 0;
        end
        
    end
end

function [codebook] = clustering(data)
% The function performs Mean-Shift vector quantization
%   @data is a matrix N*D, where N is the number of samples, D is the
%   number of features
%   @ codebook is a matrix C*D, where C is the number of discovered codebook
%   vectors

    iter_num = 100;
    sigma = 2;
    relative_change_tolerance = 0.001;
    round_decimals = 2;
    
    new_data = zeros(size(data));
    for iter_count=1:iter_num
        for obj_i=1:size(data, 1)
            % Calculate squared Euclidean distances
            distances_squared = data - repmat(data(obj_i, :), size(data, 1), 1);
            distances_squared = distances_squared .^ 2;
            distances_squared = sum(distances_squared, 2);
            % Apply kernel derivative to distances
            kernel_multipliers = kernel_derivative(distances_squared, sigma);
            normalization = sum(kernel_multipliers);
            % Broadcast multipliers to all the dimensions
            kernel_multipliers = repmat(kernel_multipliers, 1, size(data, 2));
            % Weighted sum of old data points
            new_data(obj_i, :) = sum(data .* kernel_multipliers, 1);
            
            % Normalize
            new_data(obj_i, :) = new_data(obj_i, :) / normalization;
        end
        % Compute the change in data points coordinates (max of changes over coordinate over samples)
        delta = abs(new_data - data);
        delta = max(delta(:));
        delta_normalizer = abs(data);
        delta_normalizer = max(delta_normalizer(:));
        delta = delta / delta_normalizer;
        
        % Replace old data
        data = new_data;
        
        % Check convergence
        if delta <= relative_change_tolerance
            disp(sprintf('Breaking at iteration %i with relative chage %d', iter_count, delta));
            break;
        end
    end
    
    % Aggregate data points now
    data = round(data * 10^round_decimals) / 10^round_decimals;
    codebook = unique(data, 'rows');
end

function [result] = kernel_derivative(x_squared, sigma)
% The function calculates the k'(x) for kernel k used for PDF estimation
%   It is required for each Mean-Shift step
%   @sigma is the bandwidth parameter
    if nargin < 2
        sigma = 1.0;
    end
    
    result = exp(-x_squared./sigma^2);
end

function [s, dist_id] = Sample()
    distribution = floor(3*rand());

    m = [0 0];
    sigma = [0 0];
    switch distribution
        case 0
            m = [0 0];
            sigma = [4 1];
        case 1
            m = [-10 0];
            sigma = [1 2];

        case 2
            m = [10 0];
            sigma = [1 2];
    end
    s = m+sigma.*randn(1,2);
    if distribution > 1
        dist_id = 1;
    else
        dist_id = 0;
    end
end