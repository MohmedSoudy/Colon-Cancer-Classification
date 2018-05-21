function [K_Fold_Variance, K_Fold_Error] = K_Fold_Function(Training_Set, Training_Labels, Fold_Count, Run_Count, Basyian_Flag, mkNN_Flag, SVM_Flag, All_Flag)
[Training_Count, Features_Count] = size(Training_Set);
Fold_Size = Training_Count / Fold_Count;
Error = zeros(Run_Count)
for i = 1: Run_Count
     Shuffeled_Training_Set =  Training_Set(randperm(size(Training_Set, 1)), :);
     First_Fold_Set = Shuffeled_Training_Set(1:16, :);
     Second_Fold_Set = Shuffeled_Training_Set(17:32, :);
     First_Fold_Labels =  Training_Labels(1:16, :);
     Second_Fold_Labels =  Training_Labels(17:32, :);
     if mkNN_Flag
         [Fold1_K, Fold1_Accuracy, Fold1_Conf_Matrix] = Mod_K_NN(First_Fold_Set, Second_Fold_Set, First_Fold_Labels, Second_Fold_Labels);
         [Fold2_K, Fold2_Accuracy, Fold2_Conf_Matrix] = Mod_K_NN(Second_Fold_Set, First_Fold_Set, Second_Fold_Labels, First_Fold_Labels);
     elseif Basyian_Flag
         %Basyian Classifier
         [Fold1_Normal_Set, Fold1_Cancer_Set] = Class_Split_Function(First_Fold_Set, First_Fold_Labels);
         [Fold1_Accuracy, Fold1_Conf_Matrix] = Basyian_Function(Fold1_Normal_Set, Fold1_Cancer_Set, Second_Fold_Set, Second_Fold_Labels);
         [Fold2_Normal_Set, Fold2_Cancer_Set] = Class_Split_Function(Second_Fold_Set, Second_Fold_Labels);
         [Fold2_Accuracy, Fold2_Conf_Matrix] = Basyian_Function(Fold2_Normal_Set, Fold2_Cancer_Set, First_Fold_Set, First_Fold_Labels);
     elseif SVM_Flag
         %SVM Classifier
     elseif All_Flag
          %Classifiers Majority Vote
     end
     
     Ni = Fold1_Conf_Matrix(1, 2) + Fold1_Conf_Matrix(2, 1) + Fold2_Conf_Matrix(1, 2) + Fold2_Conf_Matrix(2, 1);
     %Calculate Ni -> # of wrong samples
     Error(i) = Ni / Training_Count;
end

Sum = 0;
K_Fold_Error = sum(Error) / Run_Count;
for i =1: Run_Count
    Sum = Sum + ((Error(i) - K_Fold_Error)^2);
end
K_Fold_Variance = Sum / (Run_Count - 1);

end