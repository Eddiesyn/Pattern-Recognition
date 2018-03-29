% pr_exercise_9.m
% In this exercise we are fitting a SOM on an unknown distribution
%

function pr_exercise_9()
% iterations until convergence
t_max = 5000;
% initial learning rate
e_init = 1;
% final learning rate
e_final = .005;
% initial neighbourhood sigma
sigma_initial = 2;
% final neighbourhood sigma
sigma_final = 0.1;

% set som size
grid_size = 5;
data_value_range = [-10 10];

% three dimesional array NxNx2 to hold SOM
% 3rd dimension holds the x,y coordinates
grid=zeros(grid_size, grid_size, 2);

% initialize grid nodes randomly
for k=1:grid_size
    for l=1:grid_size
        grid(k, l, :) = (data_value_range(2)-data_value_range(1)) * (-.5+rand(1, 1, 2));
    end
end

X=zeros(t_max, 2);

for t=1:t_max
    x=Sample();
    % store sample
    X(t, :)=x; 

    % determine winner (node with smallest distance)
    min_dist=realmax;
    k_star=1; % row in grid of winner
    l_star=1; % column in grid of winner

    for k=1:grid_size
        for l=1:grid_size
            dist=sqrt( (x(1)-grid(k, l, 1)).^2 + (x(2)-grid(k, l, 2)).^2);

            if(dist<min_dist)
                min_dist=dist;
                k_star=k;
                l_star=l;
            end
        end
    end

    % update winner and neighbours
    for k=1:grid_size
        for l=1:grid_size
            delta = LearningRate(t) .* Neighbour(k-k_star, l-l_star, t);
            grid(k, l, 1) = grid(k, l, 1)+delta.*(x(1)-grid(k, l, 1));
            grid(k, l, 2) = grid(k, l, 2)+delta.*(x(2)-grid(k, l, 2));

        end
    end
end

% visualize results
plot(X(:, 1), X(:,2), 'r.', 'MarkerSize', 5);
hold on
plot(grid(:, :, 1), grid(:, :, 2), 'ko', 'LineWidth', 10);
% draw horizontal edges
for k=1:grid_size-1
    for l=1:grid_size
        plot([grid(k, l, 1), grid(k+1, l, 1)], [grid(k, l, 2), grid(k+1, l, 2)], 'LineWidth', 2);
    end
end
% draw vertical edges
for l=1:grid_size-1
    for k=1:grid_size
        plot([grid(k, l, 1), grid(k, l+1, 1)], [grid(k, l, 2), grid(k, l+1, 2)], 'LineWidth', 2);
    end
end

% Sample()
% returns a randomly picked point from the distribution that we want
% to approximate with SOM
function x=Sample()
    distribution=floor(3*rand());

    mu=[0, 0];
    sigma=[0 0];
    switch distribution
        case 0
            mu=[0 0];
            sigma=[4 1];
        case 1
            mu=[-10 0];
            sigma=[1 2];
        case 2
            mu=[10 0];
            sigma=[1 2];
    end

    x = mu'+sigma'.*randn(2, 1);
end

% LearningRate(t)
% returns the learning rate for time (iteration) t
function e = LearningRate(t)
    e=e_init * (e_final / e_init)^(t/t_max);
end

% Neighbour(delta_k, delta_l, t)
% returns the neighbour weight based on the distance from winner node (in grid space)
% and time (iteration) t
function h = Neighbour(delta_k, delta_l, t)
    sigma = sigma_initial * (sigma_final/sigma_initial).^(t/t_max);

    h = (1/(2*pi*sigma))*exp(-(delta_k.^2+delta_l.^2)/(2*sigma.^2));
end

end