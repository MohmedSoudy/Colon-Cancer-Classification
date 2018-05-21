function Variance = VarianceFeatureFn(Mean, N, X)
[rows, cols] = size(X);
Sum = zeros(1, cols);
Variance = zeros(1, cols);
for i = 1: rows
    for j = 1: cols
        Sum(1, j) = Sum(1, j) + (X(i, j) - Mean(j))^2;
    end
end

for j = 1: cols
    Variance(1, j) = Sum(1, j) /(N - 1);
end
end
