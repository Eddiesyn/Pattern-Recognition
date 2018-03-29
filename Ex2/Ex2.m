% exercise 2
% In this exercise we want to find well a binary classifier works in 
% detecting above normal values from two interleaved normal distributions.
% 1. Plot the ROC curve for for the decision rule
% 2. Plot the Precision-Recall curve of the same data
% 3. Plot the cost curve for the provided
% 4. Find the optimal point of operation

% normal distribution classes
rng(0);
numOfObservations = 1000;
mean1 = 3;
std1 = 2;
mean2 = 7;
std2= 2;
normal = sort(normrnd(mean1, std1, [1 numOfObservations]), 'descend');
aboveNormal = sort(normrnd(mean2, std2, [1 numOfObservations]), 'descend');
threshold = sort([normal aboveNormal], 'descend');

% visulaize distributions
binranges = mean1-4*std1:0.3:mean2+4*std2;
bincount1 = histc(normal, binranges);
bincount1 = bincount1/numOfObservations;
bincount1 = tsmovavg(bincount1, 's', 4, 2);

bincount2 = histc(aboveNormal, binranges);
bincount2 = bincount2/numOfObservations;
bincount2 = tsmovavg(bincount2, 's', 4, 2);
figure;
plot(binranges, bincount1);
hold on;
plot(binranges, bincount2, 'r');
hold off;
title('Distributions');
xlabel('value');
ylabel('% of samples');
legend('"normal"', '"above normal"');

% Calcualte and plot the ROC curve below
% 
% (code)
%

% Solution 1
% calculate fpr, tpr and plot roc
fp = zeros(1,0);
tp = zeros(1,0);
for i=1:2*numOfObservations
    tp = [tp sum(aboveNormal(:)>=threshold(i))];
    fp = [fp sum(normal(:)>=threshold(i))];
end
% numOfObservations is the number of both positive and negative samples
tpr = tp/numOfObservations;  % tpr = tp/[P]
fpr = fp/numOfObservations;  % fpr = fp/[N]

fid1 = figure;
plot(fpr, tpr);
title('ROC curve');
xlabel('fpr = 1 - specificity = 1 - tn/(tn+fp)');
ylabel('tpr = sensitivity = tp/(tp+fn)');

% Calculate precision, recall and plot curve below
%
% (code)
%

% Solution 2
% calculate recall, precision and plot precision-recall curve
% recall = tp/(tp+fn) = tp/[P]
% precision = tp/(tp+fp)

precision = tp./(tp+fp);
recall = tpr;

% Alternatively, one can do this:
% precision = zeros(1,0);
% for i=1:2*numOfObservations
%    precision = [precision tp(i)/(tp(i)+fp(i))];        
% end

fid2 = figure;
plot(recall, precision);
title('precision-recall curve');
xlabel('recall = sensitivity = tp/(tp+fn)');
ylabel('precision = tp/(tp+fp)');
ylim([0 1]);

% Given the follwoing costs, plot the cost function below
cfn = 1; % cost false negative
cfp = 1; % cost false positive

%
% (code)
%

% Solution 3

% fn + tp = [P] = numOfObservations
% fn = [P] - tp
cfpTotal = fp * cfp;
cfnTotal = (numOfObservations - tp) * cfn;

% Alternatively, with a separate for-loop

% cfnTotal = zeros(1,0);
% cfpTotal = zeros(1,0);
% for i=1:2*numOfObservations
%     cfnTotal = [cfnTotal cfn*(numOfObservations - tp(i))];
%     cfpTotal = [cfpTotal cfp*fp(i)];
% end
cTotal = cfnTotal + cfpTotal;

fid3 = figure;
plot(threshold, cTotal);


% Given the previous cost fuction, find the optimal threshold
%
% (code)
%

% Solution 4
[Min, index] = min(cTotal); % cTotal min index points at most the end of aboveNormal
figure(fid3);
hold on;
%plot([aboveNormal(index) aboveNormal(index)], [Min-50 max(cTotal)], 'r');
plot([threshold(index) threshold(index)], [Min-50 max(cTotal)], 'r');
hold off;

figure(fid1);
hold on;
plot([fpr(index) fpr(index)], [min(tpr) max(tpr)], 'r'); % plot vertical line
plot([min(fpr) max(fpr)], [tpr(index) tpr(index)], 'r'); % plot horizontal line