function [Accuracy, Conf_Matrix] = Basyian_Function(Training_Set_Normal, Training_Set_Cancer, Testing_Set, Actual_Labels)

[Samples_Number_Normal, Features1] = size(Training_Set_Normal);
[Samples_Number_Cancer, Features1] = size(Training_Set_Cancer);
[Test_Number, Features3] = size(Testing_Set);
Training_Samples = Samples_Number_Normal + Samples_Number_Cancer;

%CALCULATE MEAN & VARIANCE VALUE OF NORMAL CLASS
Sum_Normal = sum(Training_Set_Normal , 1);
Mean_Matrix_Normal = Sum_Normal / Samples_Number_Normal;
Variance_Normal = VarianceFeatureFn(Mean_Matrix_Normal, Samples_Number_Normal, Training_Set_Normal);

%CALCULATE MEAN & VARIANCE VALUE OF CANCER CLASS
Sum_Cancer = sum(Training_Set_Cancer , 1);
Mean_Matrix_Cancer = Sum_Cancer / Samples_Number_Cancer;
Variance_Cancer = VarianceFeatureFn(Mean_Matrix_Cancer, Samples_Number_Cancer, Training_Set_Cancer);

%CALCULATE MUs & SIGMAs OF BOTH CLASSES
Mus = [Mean_Matrix_Normal; Mean_Matrix_Cancer];
Sigmas = [Variance_Normal; Variance_Cancer];

%LIKELIHOOD VALUES FOR THE 2 CLASSES
Test_Labels = zeros(Test_Number , 1);
Features_Number = Features1;
i=1;
for k =1: Test_Number 
 for j = 1: Features_Number
     pMatrix(i, j) = mynormalfn(Testing_Set(k,j), Mus(i, j),sqrt(Sigmas(i, j)));
     pMatrix(i + 1, j) = mynormalfn(Testing_Set(k,j), Mus(i + 1, j),sqrt(Sigmas(i + 1 , j)));
 end
 
 PXw = prod(pMatrix, 2);
 
 if PXw(1) > PXw(2)
     Test_Labels(k) = 1;
 else
     Test_Labels(k) = -1;
 end
     
end
   
Conf_Matrix = Confusion_Matrix_Function(Actual_Labels, Test_Labels);
Accuracy = AccuracyFn(Conf_Matrix, Test_Number);
end
