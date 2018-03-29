% pr_exercise_13_solution.m

function pr_exercise_13_solution(index)
    if (nargin < 1)
        index = 0;
    end
    % create mesh grid
    [x,y] = meshgrid(0:0.1:30, 0:0.1:30);

    numOfSteps = size(x,1);
    x = reshape(x, [numOfSteps^2 1]);
    y = reshape(y, [numOfSteps^2 1]);

    % define reference vectors
    W1 = [1; 1];
    W2 = [3; 3];

    % define covariance matrices
    switch index
        case 1 % equal matrices diagonal
            C1 = [3 0; 0 1];
            C2 = [3 0; 0 1];
        case 2 % equal non diagonal
            C1 = [5 1; 1 3];
            C2 = [5 1; 1 3];
        case 3 % non-equal diagonal
            C1 = [3 0; 0 2];
            C2 = [2 0; 0 1];
        case 4 % non-equal, non-diagonal
            C1 = [2 0.3; 0.3 2];
            C2 = [1 0.5; 0.5 0.7];
        case 5 % non-equal non-diagonal, but diagonal elements are equal
            C1 =[3 0.3; 0.3 2];
            C2 = [3 1.9; 1.9 2];
        otherwise
            C1 = [1 0; 0 1];
            C2 = [1 0; 0 1];
    end

    C1
    C2
    
    [xW1, yW1] = CalculateArea();

    figure;
    hold on;
    plot(xW1, yW1, '.k');
    plot(W1(1), W1(2), '.b', 'MarkerSize', 30);
    plot(W2(1), W2(2), '.b', 'MarkerSize', 30);
    xlabel('x');
    ylabel('y');
    xlim([0 30]);
    ylim([0 30]);
    pbaspect([1 1 1]);


    % calculate distances and return elements that are closer to reference vector 1
    function [xW1, yW1] = CalculateArea()
        % get inverse covariance matrices
        C1inv = inv(C1);
        C2inv = inv(C2);
        labels = zeros(numOfSteps^2,1);

        for i=1:numOfSteps^2
            d1 = (W1 - [x(i); y(i)])'*C1inv*(W1 - [x(i); y(i)]);
            d2 = (W2 - [x(i); y(i)])'*C2inv*(W2 - [x(i); y(i)]);

            if (d1<d2)
                label(i) = 1;
            end
        end

        xW1 = x(label(:)==1);
        yW1 = y(label(:)==1);
    end
end