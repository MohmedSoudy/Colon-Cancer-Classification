function [K_Fold_Variance, K_Fold_Error] = K_Fold_Function(Training_Set, Training_Labels, Fold_Count, Run_Count, Basyian_Flag, mkNN_Flag, SVM_Flag, All_Flag)
[Training_Count, Features_Count] = size(Training_Set);
Fold_Size = Training_Count / Fold_Count;
Error = zeros(Run_Count, 1);

for i = 1: Run_Count
     Training_Set = [Training_Set Training_Labels];
     Shuffeled_Training_Set =  Training_Set(randperm(size(Training_Set, 1)), :);
     Training_Labels = Shuffeled_Training_Set(:, 26); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%35
     Shuffeled_Training_Set = Shuffeled_Training_Set(:, 1:25); %%%%%%%%%%%%%%%%%%%%%%%%%1:34
     First_Fold_Set = Shuffeled_Training_Set(1:16, :);        
     Second_Fold_Set = Shuffeled_Training_Set(17:32, :);       
     First_Fold_Labels =  Training_Labels(1:16, :);
     Second_Fold_Labels =  Training_Labels(17:32, :);
     if mkNN_Flag
         
         [Fold1_Normal_Set, Fold1_Cancer_Set] = Class_Split_Function(First_Fold_Set, First_Fold_Labels);
         [Fold1_K, Fold1_mkNN_Labels] = Mod_K_NN(Fold1_Cancer_Set, Fold1_Normal_Set, Second_Fold_Set ,sort(First_Fold_Labels, 'descend')', Second_Fold_Labels);
         
         [Fold2_Normal_Set, Fold2_Cancer_Set] = Class_Split_Function(Second_Fold_Set, Second_Fold_Labels);
         [Fold2_K, Fold2_mkNN_Labels] = Mod_K_NN(Fold2_Cancer_Set, Fold2_Normal_Set, First_Fold_Set ,sort(Second_Fold_Labels, 'descend')', First_Fold_Labels);
         
         [Accuracy1 , Fold1_mkNN_Conf_Matrix] = Total_Accuracy( Fold1_mkNN_Labels , Second_Fold_Labels);
         [Accuracy2 , Fold2_mkNN_Conf_Matrix] = Total_Accuracy( Fold2_mkNN_Labels , First_Fold_Labels);
         if(~Basyian_Flag && mkNN_Flag && ~SVM_Flag && ~All_Flag )
             Ni = Fold1_mkNN_Conf_Matrix(1, 2) +Fold1_mkNN_Conf_Matrix(2, 1) + Fold2_mkNN_Conf_Matrix(1, 2) + Fold2_mkNN_Conf_Matrix(2, 1);
             Error(i) = Ni / Training_Count;
         end
     end
     if Basyian_Flag
         %Basyian Classifier
         [Fold1_Normal_Set, Fold1_Cancer_Set] = Class_Split_Function(First_Fold_Set, First_Fold_Labels);
         Bayesian_Lables_1 = NaiveBayesian(Fold1_Cancer_Set ,Fold1_Normal_Set, Second_Fold_Set);
         
         [Fold2_Normal_Set, Fold2_Cancer_Set] = Class_Split_Function(Second_Fold_Set, Second_Fold_Labels);
         Bayesian_Lables_2 = NaiveBayesian(Fold2_Cancer_Set ,Fold2_Normal_Set, First_Fold_Set);
         
         [Bayes_1_Acc , Fold1_Bayes_Conf_Matrix] = Total_Accuracy( Bayesian_Lables_1 , Second_Fold_Labels ); %%%%%%%%%
         [Bayes_2_Acc , Fold2_Bayes_Conf_Matrix] = Total_Accuracy( Bayesian_Lables_2 , First_Fold_Labels ); %%%%%%%%%
         if(Basyian_Flag && ~mkNN_Flag && ~SVM_Flag && ~All_Flag )
             Ni = Fold1_Bayes_Conf_Matrix(1, 2) +Fold1_Bayes_Conf_Matrix(2, 1) + Fold2_Bayes_Conf_Matrix(1, 2) + Fold2_Bayes_Conf_Matrix(2, 1);
             Error(i) = Ni / Training_Count;
         end
     end
     if SVM_Flag
         %SVM Classifier
         [Fold1_Normal_Set, Fold1_Cancer_Set] = Class_Split_Function(First_Fold_Set, First_Fold_Labels);
         [Svm_cfit_Poly_Lables , Svm_cfit_rbf_Labels , Svm_cfit_sig_Labels ,Svm_linear_Labels , Svm_tr_rbf_Labels ,Svm_tr_poly_Labels ,Svm_tr_Sig_Labels] = SupportvectorMachine( sort(First_Fold_Labels, 'descend') , Fold1_Cancer_Set , Fold1_Normal_Set , Second_Fold_Set);
         [Fold2_Normal_Set, Fold2_Cancer_Set] = Class_Split_Function(Second_Fold_Set, Second_Fold_Labels);
         [Svm_cfit_Poly_Lables2 , Svm_cfit_rbf_Labels2 , Svm_cfit_sig_Labels2 ,Svm_linear_Labels2 , Svm_tr_rbf_Labels2 ,Svm_tr_poly_Labels2 ,Svm_tr_Sig_Labels2] = SupportvectorMachine( sort(Second_Fold_Labels, 'descend') , Fold2_Cancer_Set , Fold2_Normal_Set , First_Fold_Set);
         [Svm_cfit_sig_Accuracy , Fold1_Svmcfit1_Conf_Matrix] = Total_Accuracy( Svm_cfit_sig_Labels , Second_Fold_Labels ); %%%%%%%%%
         [Svm_cfit_sig2_Accuracy , Fold1_Svmcfit2_Conf_Matrix] = Total_Accuracy( Svm_cfit_sig_Labels2 , First_Fold_Labels ); %%%%%%%%%
        if(~Basyian_Flag && ~mkNN_Flag && SVM_Flag && ~All_Flag )
            Ni = Fold1_Svmcfit1_Conf_Matrix(1, 2) +Fold1_Svmcfit1_Conf_Matrix(2, 1) + Fold1_Svmcfit2_Conf_Matrix(1, 2) + Fold1_Svmcfit2_Conf_Matrix(2, 1);
            Error(i) = Ni / Training_Count;
        end
     end
     if All_Flag
          Majority_Pred_Labels_Fold1 = Mjority_voting(Bayesian_Lables_1 , Fold1_mkNN_Labels , Svm_cfit_sig_Labels);
          Majority_Pred_Labels_Fold2 = Mjority_voting(Bayesian_Lables_2 , Fold2_mkNN_Labels , Svm_cfit_sig_Labels2);
          [Accuracy1 , Fold1_Majority_Conf_Matrix] = Total_Accuracy(Majority_Pred_Labels_Fold1 , Second_Fold_Labels);
          [Accuracy2 , Fold2_Majority_Conf_Matrix] = Total_Accuracy(Majority_Pred_Labels_Fold2 , First_Fold_Labels);
          Ni = Fold1_Majority_Conf_Matrix(1, 2) + Fold1_Majority_Conf_Matrix(2, 1) + Fold2_Majority_Conf_Matrix(1, 2) + Fold2_Majority_Conf_Matrix(2, 1);
          Error(i) = Ni / Training_Count;
     end
end

Sum = 0;
K_Fold_Error = sum(Error) / Run_Count;
for i =1: Run_Count
    Sum = Sum + ((Error(i) - K_Fold_Error).^2);
end
K_Fold_Variance = Sum / (Run_Count - 1);
end