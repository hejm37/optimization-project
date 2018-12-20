rng('default');

% Generating X
onePosi = randi(100, 5, 1);
x_t = zeros(100, 1);
for i = 1:5
    x_t(onePosi(i)) = randn(1);
end

% Generating Y, e and caculate b
A = randn(50, 100);
e = sqrt(0.1).*randn(50, 1);

b = A*x_t+e;

filename = "variables.mat";
save(filename, 'x_t', 'A', 'e', 'b');