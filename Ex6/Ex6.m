% pr_exercise_6_solution.m
%
% In this exercise we want to find the model which best approximates an unknown Psychometric
% function. We know that the unknown Psychometric function is a displaced representation, in the
% [0,2] interval, of a basic psychometric function with threshold=4. We have 11 models whose 
% displacement varies in the [0,2] interval with a step of 0.2. There are two ways of finding 
% the best approximation.
%
% First we can get 20 samples for each small displacement (ex. dx=0.2) on x axis and calculate 
% the % "correct' for each position. Then we find the Psychometric function that fits the best 
% the salculated percentages.
%
% A more efficient approach to the problem is the use of Bayesian inferece (or learning) which 
% iteratively converges to the best model. This algorithm can be summarized 
% in the following steps:
%
% a. Assign equal weights to all models
% b. sample (get a response) at a random displacement in the [0,2] interval
% c. calculate new weights for each model (multiply old weights with the return probability of Psychometric function if sample=1 or 1-probability if sample=0)
% d. normalize new weights in order for then to sum to 1
% e. sample at the threshold of the model with the highest weight
% f. repeat from c


function pr_exercise_6_solution()

% threshold of the basic Psychometric function    
T = 4;

% displacement of the unknown Psychometric function that we want to approximate. This is
% the unknown that we want to approximate.
unknownDisp = 1.238;
%unknownDisp = 2*rand;

% displacements of the candidate models
dispStep = 0.2;
displacements = 0:dispStep:2;

% priors are initialized to be uniform (other initialization possible)
priors = ones(size(displacements))./size(displacements,2);

% probability of models for the current step. Each model represents a threshold
M = zeros(size(displacements));

% in 1st step previous model probability is equal to priors
M_n = priors;

% initial sampling in x axis is chosen randomly
c = T + rand*max(displacements);

% history of sampling points
cHist = zeros(1,0);
i=1;

% we use a fixed number of trials; alternatively,
% a convergence measure could be used for termination
while (i<200)
    % get response for current sample
    r = Response(c,T);
    % calculate new probability for each model
    for j=1:size(displacements,2)
        if (r==1)
            M(j) = M_n(end,j)*Psychometric(c,T,displacements(j));
        else
            M(j) = M_n(end,j)*(1-Psychometric(c,T,displacements(j)));
        end
    end
    % normalize. This is equivalent to 1/(\sum(P_{n-1}(M_i)* \psy_{M_i}(c)))
    M = M./sum(M);

    % add sample to history
    cHist = [cHist c];
    
    % add current models' probability to history
    M_n = [M_n; M];
    
    % calculate new sample. Equals to the threshold of the 
    % model which currently has the highest probability
    [m, index] = max(M);
    c = T + displacements(index); % basic T plus displacement of model
    i = i+1;
end 

[t, index] = sort(M_n(end,:),'descend');
fprintf('Unknown threshold %.3f Approximated threshold %.3f\n', T+unknownDisp, T+displacements(index(1)));

% visualize psychometric functions
x = 0:0.01:8;
y = Psychometric(x,T,unknownDisp);
h1 = plot(x,y, 'LineWidth', 4, 'Color', [0.2 0.2 0.7]);
hold on;
for i=1:size(displacements,2)
    y = Psychometric(x,T,displacements(i));
    if (i == index(1))
        h2 = plot(x,y, 'LineWidth', 2, 'Color', [0 0 0]);
    else
        h3 = plot(x,y, 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    end
end
xlabel('x');
ylabel('percentage "correct"');
legend([h1 h2 h3], {'unknown model', 'chosen model', 'candidate models'}, 'Location', 'northwest');

% visualize convergence
displacements = displacements+T;
x = repmat((1:size(M_n,1))',1,size(M_n,2));
y = repmat(displacements, size(M_n,1),1);
figure;
surf(x,y,M_n, 'EdgeColor', 'none');
ylabel('x');
xlabel('iteration');
zlabel('probability of each model');

% visualize sampling
binranges = T-dispStep/2:dispStep:T+2+dispStep/2;
bincounts = histc(cHist, binranges);
bincounts = bincounts./sum(bincounts);
figure;
bar(binranges, bincounts);
xlabel('x');
ylabel('percentage of samples');

%keyboard

%--------------------------------- helper functions ----------------------------
%
% Functions which provide the functionality needed for estimating the best model. Take a look
% if you want to understand the full process
   
% Returns response of unknown psychometric function for a point c in x axis 
% and basic threshold T.  r=1 "correct"
%
% input:
%   c   - position of sample
%   T   - threshold of basic psychometric function
%
% output:
%   r   - response (1 == "correct", 0 == "false")
function r = Response(c,T)
    r = 0;
    if(rand <= Psychometric(c,T,unknownDisp))
    % Probability of correct response increases as we move to higher c
        r = 1;
    end
end
   
% Returns the probability for a point c in x axis with basic threshold T
% and displacement d. If d>0 displaces to the right
% 
% input:
%   c   - position of sample 
%   T   - threshold of basic psychometric function
%   d   - displacement from basic psycometric funtion. (d>0 displaces to the right)
%
% output:
%   p   - probability of correct response for point c of the displaced by d psycometric function

function p = Psychometric(c,T,d)
    c = c-d;
    c(c(:)<0) = 0;
    p = 1 - 0.5*exp(-(c/T).^3.5);
end


end