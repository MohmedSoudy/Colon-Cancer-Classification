function [ SVMModel23, Predicted_Labels , Predicted_Labels_rbf , Predicted_Labels_sig ,SVM_Lin , SVM_rbf ,SVM_Pol ,SVM_sigm] = SupportvectorMachine( Labels_Vector , Training_Set_Cancer , Training_Set_Normal , Testing_Set)
%Train SVM & Fit generated model
%**********************using Fitcsvm********************
Feature_Matrix = [Training_Set_Normal ; Training_Set_Cancer];
SVMModel23 = fitcsvm(Feature_Matrix , Labels_Vector,'KernelFunction','linear');
SVMModel = fitcsvm(Feature_Matrix , Labels_Vector,'KernelFunction','polynomial');
SVM_Second_Model = fitcsvm(Feature_Matrix , Labels_Vector,'KernelFunction','rbf');
SVM_Sigmoid_Model = fitcsvm(Feature_Matrix , Labels_Vector,'KernelFunction','sigmoid');
[Predicted_Labels,score] = predict(SVMModel,Testing_Set);
[Predicted_Labels_rbf,score_rbf] = predict(SVM_Second_Model,Testing_Set);
[Predicted_Labels_sig,score_sig] = predict(SVM_Sigmoid_Model,Testing_Set);
%********************using SvmTrain & SvmClassify*******************
SVM_linear = svmtrain(Feature_Matrix , Labels_Vector ,'kernel_function','linear');
SVM_Lin = svmclassify(SVM_linear , Testing_Set);
SVM_Poly = svmtrain(Feature_Matrix , Labels_Vector ,'kernel_function','polynomial');
SVM_Pol = svmclassify(SVM_Poly , Testing_Set);
SVM_RBF = svmtrain(Feature_Matrix , Labels_Vector ,'kernel_function','rbf');
SVM_rbf = svmclassify(SVM_RBF , Testing_Set);
SVM_sigmo = svmtrain(Feature_Matrix , Labels_Vector ,'kernel_function' ,@(u,v)sigmoid(u,v));
SVM_sigm = svmclassify(SVM_sigmo , Testing_Set);
end
