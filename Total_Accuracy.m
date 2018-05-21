function [ OVerall_Accuracy , Confusion_Matrix ] = Total_Accuracy( Label_Vector , Predicted_Labels )
%Calculate Accuracy from labels , labels generated from classifier
[N_of_Samples , N_of_Coloumns] = size(Label_Vector);
Number_Of_Classes = 2;
Confusion_Matrix = zeros(Number_Of_Classes,Number_Of_Classes);
%***********************Overall_Accuracy***********************************
Counter = 0;
for i = 1 :N_of_Samples
    if (Label_Vector(i) == Predicted_Labels(i))
        Counter = Counter + 1;
    end
    if (Label_Vector(i) == -1 && Predicted_Labels(i) == -1)
        Confusion_Matrix(1,1) = Confusion_Matrix(1,1) + 1;
    elseif(Label_Vector(i) == 1 && Predicted_Labels(i) == 1)
        Confusion_Matrix(2,2) = Confusion_Matrix(2,2) + 1;
    elseif(Label_Vector(i) == -1 && Predicted_Labels(i) == 1)
        Confusion_Matrix(1,2) = Confusion_Matrix(1,2) + 1;
    elseif(Label_Vector(i) == 1 && Predicted_Labels(i) == -1)
        Confusion_Matrix(2,1) = Confusion_Matrix(2,1) + 1;
    end
end
OVerall_Accuracy = (Counter/N_of_Samples)*100;
end
