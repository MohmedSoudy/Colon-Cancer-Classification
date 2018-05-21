Data = dlmread('C:\Users\Dell\Desktop\Pattern Project\ColonCancer_PCA.txt');
Data_Complete = dlmread('C:\Users\Dell\Desktop\Pattern Project\ColonCancer_Features-1.txt');
Labels = dlmread('C:\Users\Dell\Desktop\Pattern Project\ColonCancerLabels.txt');
Training_Set = Data(1:32, :);
Test_Set = Data(33:62, :);
Training_Labels = Labels(:, 1:32);
Test_Labels = Labels(:, 33:62);

[K, mkNN_Accuracy, mkNN_Conf_Matrix] = Mod_K_NN(Training_Set, Test_Set, Training_Labels, Test_Labels);
[Normal_Set, Cancer_Set] = Class_Split_Function(Training_Set, Training_Labels);
[Basyian_Accuracy, Basyian_Conf_Matrix] = Basyian_Function(Normal_Set, Cancer_Set, Test_Set, Test_Labels);

%[Variance, Error] = K_Fold_Function(Training_Set, Training_Labels, 2, 3);