Feature_Matrix = dlmread('ColonCancer_Features.txt');
X = PCA(Feature_Matrix);
[y,score] = pca(Feature_Matrix);
dlmwrite('ColonCancer_PCA.txt' , score);