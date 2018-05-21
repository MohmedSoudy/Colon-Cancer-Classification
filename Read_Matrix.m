function [ Labels , Features ] = Read_Matrix(Labels_Vector , Feature_Matrix)
%Preprocessing of Labels & Features
[Labels, indeces] = sort(Labels_Vector, 'descend');
Features = Feature_Matrix(indeces, :);
end

