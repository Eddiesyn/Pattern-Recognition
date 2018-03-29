function pr_exercise_10_visualization()
% last point added as noise point
points=[1 1.7;   % x0
        2.2 1.4; % x1
        2.1 2.3; % x2
        0.7 2.2; % x3
        2.6 2.9; % x4
        1.5 4];  % x5

length = size(points,1);
distances = zeros(length);
for i=1:length
    for j=1:length
        distances(i,j) = sqrt((points(i,1)-points(j,1)).^2 + (points(i,2)-points(j,2))^2);
    end
end

distances

radius = 1.5;
minPts = 2;

fig = figure;
plot(points(:,1),points(:,2), 'r.', 'LineWidth', 3);
names= {'x0', 'x1', 'x2', 'x3', 'x4', 'x5'};

% get point type
distances = distances + diag(1:size(distances))*1000; % assign high distances to diagonal elements

% get neighbours
neighbours = distances<=radius;

% assign points
corePoints = sum(neighbours,2)>=minPts;

hold on;
for i=1:length
    [x,y] = CreateCircle(points(i,:), radius);

    if (corePoints(i))
        hCore = plot(x,y, 'g', 'LineWidth', 1); % green
    else
        % count neighbour core points
        coreNeighbours = sum(corePoints(neighbours(i,:)));
        if (coreNeighbours>0) % border point
            hBorder = plot(x,y, 'b', 'LineWidth', 1); % blue
        else % outlier
            hOutlier = plot(x,y, 'r', 'LineWidth', 1); % red
        end 
    end
end

text(points(:,1), points(:,2), names);
axis('square');
xlim([-1 6]);
ylim([-1 6]);
legend([hCore, hBorder, hOutlier], {'core point', 'border point', 'outlier'}, 'Location', 'northeast');
set(gca,'position',[0 0.05 1 0.9333],'units','normalized')
saveas(fig, 'figures/exercise10.png');


function [x, y] = CreateCircle(center, radius)
    l = linspace(0, 2*pi, 100);
    x = radius.*cos(l) + center(1);
    y = radius.*sin(l) + center(2);
end

end