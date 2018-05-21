[ Label_vector , Feature_Matrix ] = read_dataset('G:\mohmed\Fourth year\First term\Pattern\Labs\Pattern Project\colon-cancer dataset.txt');
% Features1 = Feature_Matrix;
new = PCA_task(Feature_Matrix,Label_vector);
Features1 = dlmread('Less_Features.txt');
Features1 = Features1(:,1 :34);
Labels1 = dlmread('ColonCancerLabels.txt');
Processed_data = load('Selectedvectors.mat');
Test = load('Testset.mat');
Test = Test.TestSet;
Processed_data = Processed_data.eign_vectors_Selected;
eign_val = Processed_data(:,1:34);
Mean_data = load('Mean_Matrix.mat');
Mean_data = Mean_data.Mean_Matrix;
Mean_Substracted = bsxfun(@minus,Test(1),Mean_data);
Mean_Substracted1 = bsxfun(@minus,Test(2),Mean_data);
Mean_Substracted2 = bsxfun(@minus,Test(3),Mean_data);
Test1 = Mean_Substracted*eign_val;
Test2 = Mean_Substracted1*eign_val;
Test3 = Mean_Substracted1*eign_val;
TestSet =[Test1;Test2];
TestSet = [TestSet; Test3];
[New_Labels , New_Matrix] = Read_Matrix(Labels1,Features1);
[Training_Set_Cancer , Training_Set_Normal  ,Testing_Set  ,Label_Vector , Testing_Label] = Data_Preprocessing( New_Matrix , New_Labels');
[Bayes_Labels, Mus, Sigmas] = NaiveBayesian(Training_Set_Cancer , Training_Set_Normal , Testing_Set);
[Bayes_Accuracy , Bayes_Confusion_Matrix] = Total_Accuracy(Testing_Label , Bayes_Labels);
[Model, Poly_Labels , rbf_Labels , sig_Labels , Lin , rbf , poly , sigm] = SupportvectorMachine( Label_Vector , Training_Set_Cancer , Training_Set_Normal , Testing_Set);
[Svm_Cfit_Poly_Accuracy , poly_Confusion_Matrix] = Total_Accuracy(Testing_Label , Poly_Labels);
[Svm_Cfit_rbf_Accuracy , rbf_Confusion_Matrix] = Total_Accuracy(Testing_Label , rbf_Labels);
[Svm_Cfit_sig_Accuracy , sig_Confusion_Matrix] = Total_Accuracy(Testing_Label , sig_Labels);
[Svm_Linear_Accuracy , Linear_Confusion_Matrix] = Total_Accuracy(Testing_Label , Lin);
[Svm_Tr_rbf_Accuracy , R_Confusion_Matrix] = Total_Accuracy(Testing_Label , rbf);
[Svm_Tr_poly_Accuracy ,P_Confusion_Matrix] = Total_Accuracy(Testing_Label , poly);
[Svm_Tr_Sig_Accuracy , S_Confusion_Matrix] = Total_Accuracy(Testing_Label , sigm);
[k, Mknn_Labels] = Mod_K_NN(Training_Set_Cancer, Training_Set_Normal,Testing_Set, Label_Vector', Testing_Label);
[Mknn_Accuracy , Mknn_Confusion_Matrix] = Total_Accuracy(Testing_Label , Mknn_Labels);
Majority_Labels = Mjority_voting(Bayes_Labels ,Mknn_Labels , sig_Labels);
[Majority_Accuracy , Majority_Conf_Matrix] = Total_Accuracy(Testing_Label , Majority_Labels');
Training_Set = [Training_Set_Normal ; Training_Set_Cancer];
% [K_Fold_Variance_Bayes, K_Fold_Error_Bayes] = K_Fold_Function(Training_Set, Label_Vector, 2, 3, 1, 0, 0, 0);
% [K_Fold_Variance_mKNN, K_Fold_Error_mKNN] = K_Fold_Function(Training_Set, Label_Vector, 2, 3, 0, 1, 0, 0);
% [K_Fold_Variance_SVM, K_Fold_Error_SVM] = K_Fold_Function(Training_Set, Label_Vector, 2, 3, 0, 0, 1, 0);

[SVM_LABELS, score_sig] = predict(Model,TestSet);
MKNN_LABELS = mKNN_Test(TestSet, k, Training_Set_Cancer, Training_Set_Normal, Label_Vector');
BAYES_LABELS = NaiveBayesian_Test(Mus, Sigmas, TestSet, 3, 34);
FINAL_TEST_LABELS = Mjority_voting(BAYES_LABELS, MKNN_LABELS, SVM_LABELS);
% [K_Fold_Variance_Majority, K_Fold_Error_Majority] = K_Fold_Function(Training_Set, Label_Vector, 2, 3, 1, 1, 1, 1);