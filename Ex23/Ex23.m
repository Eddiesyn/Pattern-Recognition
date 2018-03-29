% pr_exercise_23_solution.m

function pr_exercise_23_solution()
    % data creation and splitting
        % set the seed
        rng(0);

        numOfSamples = 1000;
        samples = zeros(numOfSamples,2);
        labels = zeros(numOfSamples,1);

        for i=1:numOfSamples
            [samples(i,:), labels(i)] = Sample();
        end

        % add an extra coordinate (always = 1) to each vector for further use 
        % with the perceptron
        samples = cat(2, ones(size(samples, 1), 1), samples);
        % split data into training and testing sets
        % specifying the percentage of vectors in each data set part
        % and the respective id for each part of the data set
        trainShare = 0.7;
        train_id = 1;

        % just 2 parts, so testingShare = 1 - trainShare 
        testingShare = 1 - trainShare;
        testing_id = 2;

        % create a multinomial distribution with the above probabilities
        splitDistribution = makedist('Multinomial', 'probabilities', [trainShare, testingShare]);
        % get a sample of the distribution
        indices = random(splitDistribution, size(labels));

        trainIndices = indices == train_id;
        testIndices = indices == testing_id;

        % split data and labels to training testing and testing subsets
        trainCoord = samples(trainIndices,:);
        trainLabels = labels(trainIndices);
        testCoord = samples(testIndices,:);
        testLabels = labels(testIndices);
    % done with data creation and splitting
        
    % Initialize weights with random values. 
    weights = rand(size(trainCoord, 2), 1);
    %weights = zeros(1,3);
    
    hasConverged = false;
    iteration = 1;
    misclassified = zeros(1,0);
    

    while (~hasConverged)
        hasConverged = true;
        misclassified = [misclassified 0];
        for i=1:size(trainCoord,1)
            % get predicted label by using perceptron
            predictedLabel = sign(weights' * trainCoord(i, :)');

            % for linearly separable data, perceptron always converges
            if (predictedLabel ~= trainLabels(i))
                hasConverged = false;
                misclassified(iteration) = misclassified(iteration) + 1;
            end

            % update weights
            delta = (trainLabels(i) - predictedLabel);
            weights = weights + delta * trainCoord(i, :)';
        end

        iteration = iteration+1;
    end

    % visualize samples from training set
    figure;
    hold on;
    gscatter(trainCoord(:,2), trainCoord(:,3), trainLabels, 'rb', '.');
    legend({'class -1 training set', 'class +1 training set'}, 'Location', 'southeast');
    axis square;
    xlim([-10 15]);
    ylim([-10 15]);
    % visualize perceptron
    x=-10:0.1:15;
    b = trainCoord(1, 1);
    y= -(weights(2)*x + weights(1) * b)/weights(3);
    x = x(y(:)>-10 & y(:)<15);
    y = y(y(:)>-10 & y(:)<15);
    plot(x,y,'k', 'LineWidth', 3);
    
     % visualize samples from test data set
    figure;
    hold on;
    gscatter(testCoord(:,2), testCoord(:,3), testLabels, 'rb', '.');
    legend({'class -1 test set', 'class +1 test set'}, 'Location', 'southeast');
    axis square;
    xlim([-10 15]);
    ylim([-10 15]);
    % visualize perceptron again (don;t need to calculate x and y again)
    plot(x,y,'k', 'LineWidth', 3);

    % check the quality of the classifier
    predictedLabels = zeros(size(testLabels));
    % perceptron
    predictedLabels(:) = sign(weights' * testCoord');

    % since we haven't defined positive class we'll check only the accuracy
    accuracy = sum(predictedLabels==testLabels)/size(testLabels,1);
    accuracy
end

% sampling function
% 
function [coordinates, label] = Sample()
    distribution = floor(2*rand());

    coordinates = zeros(1,2);
    label = 0;

    switch distribution
        case 0
            m = [0 0];
            sigma = [2 2];
            coordinates(1) = m(1) + sigma(1)*randn();
            coordinates(2) = m(2) + sigma(2)*randn();
            label = -1;
        case 1
            m = [8 7];
            sigma = [1 1];
            coordinates(1) = m(1) + sigma(1)*randn();
            coordinates(2) = m(2) + sigma(2)*randn();
            label = 1;
    end

end