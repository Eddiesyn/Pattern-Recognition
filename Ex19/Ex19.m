% pr_exercise_19_solution.m

function pr_exercise_19_solution()

global svm_kernel
svm_kernel = 'rbf';
%set seed for random number generators    
rng(0);

numOfSamples = 1000;
samples = zeros(numOfSamples,2);
labels = zeros(numOfSamples,1);

for i=1:numOfSamples
    [samples(i,:), labels(i)] = Sample();
end

% split data into training, validation and testing
% specifying the percentage of points in each data set part
% 
trainShare = 0.5;
train_id = 1;

validationShare = 0.3;
validation_id = 2;

testShare = 0.2;
test_id = 3;

% create a multinomial distribution with the above probabilities
splitDistribution = makedist('Multinomial', 'probabilities', [trainShare, validationShare, testShare]);
% get a sample of the distribution
indices = random(splitDistribution, size(labels));

trainIndices = indices == train_id;
validIndices = indices == validation_id;
testIndices = indices == test_id;

% split data and labels to training validation and testing subsets
trainCoord = samples(trainIndices,:);
trainLabels = labels(trainIndices);
validCoord = samples(validIndices,:);
validLabels = labels(validIndices);
testCoord = samples(testIndices,:);
testLabels = labels(testIndices);

% visualize samples
figure;
gscatter(trainCoord(:,1),trainCoord(:,2),trainLabels,'rb','.');
legend({'0-negative', '1-positive'});
title('training data');


% iterate through different c
c = -7:0.25:7;
f1Valid = zeros(size(c));
f1Train = zeros(size(c));
for i=1:size(c,2)
    c(i) = exp(c(i));
    % train svm
    svmStruct = fitcsvm(trainCoord, trainLabels, 'KernelFunction', svm_kernel, 'BoxConstraint', c(i), 'ClassNames', [0,1]);

    f1Train(i) = Validate(svmStruct, trainCoord, trainLabels);
    f1Valid(i) = Validate(svmStruct, validCoord, validLabels);
end

% find best fit to validation data
[m, index] = max(f1Valid);
% get the best SVM model
svmStruct = fitcsvm(trainCoord, trainLabels, 'KernelFunction', svm_kernel, 'BoxConstraint', c(index), 'ClassNames', [0,1]);
f1Test = Validate(svmStruct, testCoord, testLabels);

% visualize f1 scores
% C-axis is in logarithmic units
figure;
hold on;
plot(log(c), f1Train, 'r');
plot(log(c), f1Valid, 'b');
% plot a horizontal line
plot([log(min(c)) log(max(c))], [f1Test f1Test], '-.k');
% visualize the best C parameter
plot([log(c(index)) log(c(index))], [min([f1Valid 0.5]) 1], 'k');
legend({'f1 score training', 'f1 score validation', 'f1 score testing for the best model', 'chosen C'}, 'Location', 'southeast');
xlabel('log C');
ylabel('F1 score');
xlim([-7, 7])
ylim([min([f1Valid f1Test f1Train 0.5]), 1]);

PlotFitting(c(index), trainCoord, trainLabels, trainCoord, trainLabels);
legend({'0-negative training', '1-positive training', 'training decision border'}, 'Location', 'southwest');
title('training data with decision border');
PlotFitting(c(index), trainCoord, trainLabels, testCoord, testLabels);
legend({'0-negative testing', '1-positive testing', 'training decision border'}, 'Location', 'southwest');
title('test data with decision border');



end

% plotting decision border from training data and visualize with test data
function PlotFitting(c, trainCoord, trainLabels, testCoord, testLabels)
    global svm_kernel
    svmStruct = fitcsvm(trainCoord, trainLabels, 'KernelFunction', svm_kernel, 'BoxConstraint', c, 'ClassNames', [0,1]);

    figure;
    hold on;
    d = 0.1;
    border = 2.5;
    [x1Grid,x2Grid] = meshgrid(min(testCoord(:,1))-border:d:max(testCoord(:,1)+border), min(testCoord(:,2))-border:d:max(testCoord(:,2))+border);
    xGrid = [x1Grid(:),x2Grid(:)];
    [~,scores] = predict(svmStruct,xGrid);
    h(1:2) = gscatter(testCoord(:,1),testCoord(:,2),testLabels,'rb','.');
    contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');

end

% validation function. @labels are the correct class labels for comparison
function [f1Score] = Validate(svmStruct, coord, labels)
    % classify samples with svm
    svmLabels = zeros(size(labels));
    svmLabels = predict(svmStruct, coord);

    % calculate TP, FP, FN
    TP = sum(svmLabels==labels & svmLabels==1);
    FP = sum(svmLabels~=labels & svmLabels==1);
    FN = sum(svmLabels~=labels & svmLabels==0);

    % calculate precision, recall
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);

    % set F1 score
    %b = 1;
    %f1score = (1+b^2)*precision*recall/(b^2*precision+recall);
    if TP ~= 0
        f1Score = 2*precision*recall/(precision+recall);
    else
        f1Score = 0.0;
    end
end

% sampling function
% label = 1 is the "positive"
function [coordinates, label] = Sample()
    distribution = floor(4*rand());

    coordinates = zeros(1,2);
    label = 0;

    switch distribution
        case 0
            m = [0 0];
            sigma = [2 2];
            coordinates(1) = m(1) + sigma(1)*randn();
            coordinates(2) = m(2) + sigma(2)*randn();
            label = 1;
        case 1
            m = [-7 7];
            sigma = [2 2];
            coordinates(1) = m(1) + sigma(1)*randn();
            coordinates(2) = m(2) + sigma(2)*randn();
            label = 0;
        case 2
            m = [3 7];
            sigma = [2 2];
            coordinates(1) = m(1) + sigma(1)*randn();
            coordinates(2) = m(2) + sigma(2)*randn();
            label = 1;
        case 3
            m = [7 0];
            sigma = [2 2];
            coordinates(1) = m(1) + sigma(1)*randn();
            coordinates(2) = m(2) + sigma(2)*randn();
            label = 0;
    end

end
