% pr_exercise_3_solution.m

% Read again the sescirption and answer the following:
% 
% 1. Based on your observations are all the selected featured needed for correct classifications?
%
% 2. Based on the assumptiom that the selected features are conditionally independent classify the observations
%
% 3. Visualize the risk of a wrong decision for observetion 5 by varying the coordinates under the assumption of a Zero-One loss model

% load data
observations = importdata('pr_exercise_3_observations.txt', ',', 1);

colheaders = observations.colheaders;
data = observations.data;
% Task 1
%
% (code)
%
featureMean = zeros(0,5);
featureStd = zeros(0,5);
for id=unique(data(:,1))'
    featureMean = [featureMean; mean(data(data(:,1)==id,2)) mean(data(data(:,1)==id,3)) mean(data(data(:,1)==id,4)) mean(data(data(:,1)==id,5)) mean(data(data(:,1)==id,6))];
    featureStd = [featureStd; std(data(data(:,1)==id,2)) std(data(data(:,1)==id,3)) std(data(data(:,1)==id,4)) std(data(data(:,1)==id,5)) std(data(data(:,1)==id,6))];
end

% display values
featureMean
featureStd

% create histograms for each feature
for featureId = 2:size(colheaders,2)
    xMinLim = min(data(:,featureId));
    xMaxLim = max(data(:,featureId));
    figure;
    for id=unique(data(:,1))'
        subplot(2,2,id)
        hist(data(data(:,1)==id,featureId), 32);
        title([colheaders{1,featureId} ' levels for disease with id ' num2str(id)]);
        xlim([xMinLim xMaxLim]);
    end
end


% Task 2
%
% (code)
%

% calculate priors
priors = [0.4 0.1 0.3 0.2];

% since the tiredness level varies from 0 to 10 we will use the histogram as 
% approximation to te distribution of samples. Each bin will have a range of 0.5
% Headache levels are out since they follow the same distribution.
% The rest of the attributes will be handles as normal distributions
binranges = 0:0.5:10;
pdfs = zeros(0, size(binranges,2));
for id=unique(data(:,1))'
    bincounts = histc(data(data(:,1)==id,5), binranges);
    bincounts = bincounts./sum(bincounts);
    pdfs = [pdfs; bincounts'];
end

% observations (latitude, longitude, bodyTemperature, tiredness, headache)
observations = [42.12 10.43 37.7 5.0 2.1;
                49.82 22.89 38.6 3.4 3.9;
                58.90 18.71 40.1 2.2 7.7;
                32.85 06.35 38.2 8.6 1.0;
                54.56 22.43 38.3 9.3 9.0;
                42.85 -0.35 36.2 1.6 4.2];

% rows = observations
% columns = disease 
probabilities = zeros(size(observations,1), size(featureMean,1));

for obsId=1:size(observations,1)
    for disId=1:size(featureMean,1)
        probLat = normpdf(observations(obsId,1), featureMean(disId,1), featureStd(disId,1));
        probLong = normpdf(observations(obsId,2), featureMean(disId,2), featureStd(disId,2));
        probTemp = normpdf(observations(obsId,3), featureMean(disId,3), featureStd(disId,3));
        probTir = pdfs(disId, floor(observations(obsId,4)*2)+1);
        probabilities(obsId, disId) = priors(disId)*probLat*probLong*probTemp*probTir;
    end
end

[m,diseaseId] = max(probabilities,[],2);

probabilities
diseaseId

% Task 3
%
% (code)
%

risk = zeros(100,100);
x = zeros(100,100);
y = zeros(100,100);
obsId=1;
otherIds = [1:diseaseId(obsId)-1 diseaseId(obsId)+1:size(featureStd,1)];
for i=1:100
    lat = i;
    for j=1:100
        long = j-30;
        for disId=otherIds
            probLat = normpdf(lat, featureMean(disId,1), featureStd(disId,1));
            probLong = normpdf(long, featureMean(disId,2), featureStd(disId,2));
            probTemp = normpdf(observations(obsId,3), featureMean(disId,3), featureStd(disId,3));
            probTir = pdfs(disId, floor(observations(obsId,4)*2)+1);

            risk(i,j) = risk(i,j) + priors(disId)*probLat*probLong*probTemp*probTir;
        end
        x(i,j) = long;
        y(i,j) = lat;
    end
end

% normalize risk
%risk(:,:) = risk(:,:)./max(max(risk(:,:)));

% plot risk
figure;
surf(x,y,risk, 'EdgeColor', 'none');
title('Risk of wrong decision');
xlabel('longitude');
ylabel('latitude');


decProb = zeros(100,100);
x = zeros(100,100);
y = zeros(100,100);
obsId=1;
disId = diseaseId(obsId);
for i=1:100
    lat = i;
    for j=1:100
        long = j-30;
        probLat = normpdf(lat, featureMean(disId,1), featureStd(disId,1));
        probLong = normpdf(long, featureMean(disId,2), featureStd(disId,2));
        probTemp = normpdf(observations(obsId,3), featureMean(disId,3), featureStd(disId,3));
        probTir = pdfs(disId, floor(observations(obsId,4)*2)+1);

        decProb(i,j) = priors(disId)*probLat*probLong*probTemp*probTir;
        x(i,j) = long;
        y(i,j) = lat;
    end
end

% normalize risk
%decProb(:,:) = decProb(:,:)./max(max(decProb(:,:)));

% plot risk
figure;
surf(x,y,decProb, 'EdgeColor', 'none');
title('Probability of x given class');
xlabel('longitude');
ylabel('latitude');