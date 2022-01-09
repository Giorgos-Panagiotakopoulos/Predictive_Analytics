function singlesmoothing()
 
data = [18,18.2,18.24,18.41,18.32,18.6,18.57,18.81,18.9,19.1,19.2,19.23,19.3,19.15,19.22,19.8,20,19.9,19.7,19.5,19.64,19.55,19.32,19.28,19.27,19.29,19.31];
rand('state', 4); % to get the same sequence of random numbers each time
data = data + 0.7*(2*rand(size(data))-1);
time = 1:length(data);
 
n = length(data);
s1 = singleSmoothed(data, 0.1);
s2 = singleSmoothed(data, 0.5);
s3 = singleSmoothed(data, 0.9);
 
mse1 = meanSquaredError(data(2:n), s1(2:n));
mse2 = meanSquaredError(data(2:n), s2(2:n));
mse3 = meanSquaredError(data(2:n), s3(2:n));
 
figure;
plot(time, data, '-sk', 'MarkerFaceColor', 'g');
hold on;
 
plot(time(2:n), s1(2:n), '-r');
plot(time(2:n), s2(2:n), '-b', 'LineWidth', 2);
plot(time(2:n), s3(2:n), '-m');
 
legend('Original data',...
    ['alpha=0.1, mse=' num2str(mse1)],...
    ['alpha=0.5, mse=' num2str(mse2)],...
    ['alpha=0.9, mse=' num2str(mse3)],...
    'Location', 'SouthWest');
title('Single exponential smoothing');
xlabel('Time (days)');
ylabel('Water temperature (degrees C)');
 
mse = [];
alpha=0:0.05:1;
for a=alpha
    s = singleSmoothed(data, a);
    mse = [mse meanSquaredError(data(2:n), s(2:n))];
end
figure;
plot(alpha, mse);
title('Mean squared error versus alpha');
xlabel('alpha');
ylabel('MSE');
 
% Bootstrapping of forecasts
% We use the last data point and the last smoothed point to calculate
% forecasts.
 
N = 20; % number of forecasts
alpha = 0.5; % best alpha (lowest MSE)
yorigin = data(n);
f = zeros(1,N);
f(1) = s2(n); % smoothed value at origin
for i = 1:N-1
    f(i+1) = alpha*yorigin + (1-alpha)*f(i);
end
 
figure;
plot(time, data, '-sk', 'MarkerFaceColor', 'g');
hold on;
plot(time(2:n), s2(2:n), '-b');
plot([n:n+N-1], f, '-b', 'LineWidth', 3);
title('Bootstrapping of forecasts');
ylabel('Water temperature (degrees C)');
xlabel('Time (days)');
legend('Observed data', 'Smoothed data', 'Forecast');
end
 
function s = singleSmoothed(data, alpha)
% Calculates single exponentially smoothed data with weight parameter
 
n = length(data);
s = zeros(1,n+1);
s(2) = data(1);
for i = 3:n+1
    s(i) = alpha*data(i-1) + (1-alpha)*s(i-1);
end
end
 
function mse = meanSquaredError(x,y)
% Calculates the mse between two vectors
 
mse = mean((x-y).*(x-y));
end
