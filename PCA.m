function Features  = PCA(Feature_Matrix)
%Apply Principal component analysis on data to get Most correlated features
[N_of_Samples , N_of_Features] = size(Feature_Matrix);
%Get Each Coloumn Sum
Sum = sum(Feature_Matrix,1);
Mean_Matrix = Sum / N_of_Samples;
Mean_Substracted = bsxfun(@minus,Feature_Matrix,Mean_Matrix);
Covariance_Matrix = zeros(N_of_Features, N_of_Features);
for i = 1:N_of_Features
    for j = 1:N_of_Features
        Temp = Mean_Substracted(:, i) .* Mean_Substracted(:, j);
        Summation = sum(Temp);
        Covariance_Matrix(i,j) = Summation / (N_of_Samples - 1);
    end
end
[eign_vectors, eign_values] = eig(Covariance_Matrix);
eigenValues = diag(eign_values);
[sorted_values, indeces] = sort(eigenValues, 'descend');
eign_vectors = eign_vectors(:, indeces);
%Extracting Important Eigen Vectors
Normalization_Matrix = sorted_values / sum(sorted_values) *100;
PC_Sum = cumsum(Normalization_Matrix);
Pc_Index = 0;
for k = 1 : N_of_Features
    if (PC_Sum(k) >= 90)
        Pc_Index = k;
        break;
    end
end
eign_vectors_Selected = zeros(N_of_Features ,Pc_Index);
eign_vectors_Selected = eign_vectors(:,1:Pc_Index);
Features = zeros(N_of_Samples , Pc_Index);
Features = eign_vectors_Selected' * Mean_Substracted';
Features = Features';
end