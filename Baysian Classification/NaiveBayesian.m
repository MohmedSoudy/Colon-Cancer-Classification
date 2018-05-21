function [Predicted_Labels, Mus, Sigmas] = NaiveBayesian(Training_Set_Cancer , Training_Set_Normal , Testing_Set)

[Test_Number , N_of_Features_Test] = size(Testing_Set);

% Get # of Samples , # of Classes 
[Samples_Number_Cancer , N_of_Features_Cancer] = size(Training_Set_Cancer);
[Samples_Number_Normal , N_of_Features_Normal] = size(Training_Set_Normal);

%CALCULATE MEAN & VARIANCE VALUE OF CANCER CLASS
Sum_Cancer = sum(Training_Set_Cancer , 1);
Mean_Matrix_Cancer = Sum_Cancer / Samples_Number_Cancer;
Mean_Substracted_Cancer = bsxfun(@minus, Training_Set_Cancer, Mean_Matrix_Cancer);
Variance_Cancer = VarianceFeatureFn(Mean_Matrix_Cancer, Samples_Number_Cancer, Training_Set_Cancer);

%CALCULATE MEAN & VARIANCE VALUE OF NORMAL CLASS
Sum_Normal = sum(Training_Set_Normal , 1);
Mean_Matrix_Normal = Sum_Normal / Samples_Number_Normal;
Mean_Substracted_Normal = bsxfun(@minus, Training_Set_Normal, Mean_Matrix_Normal);
Variance_Normal = VarianceFeatureFn(Mean_Matrix_Normal, Samples_Number_Normal, Training_Set_Normal);

%CALCULATE MUs & SIGMAs OF BOTH CLASSES
Mus = [Mean_Matrix_Cancer; Mean_Matrix_Normal];
Sigmas = [Variance_Cancer; Variance_Normal];

%LIKELIHOOD VALUES FOR THE 2 CLASSES
Predicted_Labels = zeros(Test_Number , 1);
i=1;
for k =1:Test_Number 
 for j = 1:N_of_Features_Cancer
     pMatrix(i, j) = mynormalfn(Testing_Set(k,j), Mus(i, j),sqrt(Sigmas(i, j)));
     pMatrix(i +1, j) = mynormalfn(Testing_Set(k,j), Mus(i + 1, j),sqrt(Sigmas(i + 1 , j)));
 end
 PXw = prod(pMatrix, 2);

 if PXw(1) >= PXw(2)
     Predicted_Labels(k)= -1;
 else
     Predicted_Labels(k)= 1;
 end  
end
end


