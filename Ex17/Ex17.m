% pr_exercise_17_solution.m

function pr_exercise_17_solution()
    % Create mesh grid
    [x,y] = meshgrid(-10:0.1:10, -10:0.1:10);
    delta = zeros(size(x));
    
    % YOUR CODE HERE
    A = [1, 1.5; 1.5, 0];
    b = [0; -2];
    c = 4;
    
    numOfSteps = size(x,1);
    for x_ind=1:numOfSteps
        for y_ind=1:numOfSteps
            point = [x(x_ind, y_ind); y(x_ind, y_ind)];
            % Compute decision function
            % YOUR CODE HERE
            % delta(i, j) = ...
            delta(x_ind, y_ind) = point' * A * point + point' * b + c;
            
            % DEBUG: validate matrix form equivalency to equation form
            if abs(delta(x_ind, y_ind) - (3 * point(1) * point(2) + point(1)^2 - 2 * point(2) + 4)) > 1e-10
                disp('WRONG!')
                break
            end
        end
    end

    figure;
    hold on;
    % visualize decision function
    mesh(x, y, delta);
    % visualize delta = 0
    % last parameter for color
    mesh(x, y, zeros(size(x)), -500*ones(size(x)));
    xlabel('x_1');
    ylabel('y_2');
    zlabel('delta')
    xlim([-10 10]);
    ylim([-10 10]);
end