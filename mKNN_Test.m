function y_pred = mKNN_Test(Test_Set, k, Training_Set_Cancer, Training_Set_Normal, Training_Labels)
Training_Set = [Training_Set_Normal; Training_Set_Cancer];

[Samples_Count, Features_Count] = size(Training_Set);
[Test_Samples_Count, Test_Features_Count] = size(Test_Set);
Distance_Matrix = zeros(Samples_Count, Test_Samples_Count);

Max_Vector = max(Training_Set);
for i = 1 : Samples_Count
    Training_Set(i, :) = Training_Set(i, :) ./ Max_Vector(1,1); 
    if(i <= Test_Samples_Count)
        Test_Set(i, :) = Test_Set(i, :) ./ Max_Vector(1,1); 
    end
end

for i = 1 : Samples_Count
    for j = 1 : Test_Samples_Count
     Distance_Matrix(i, j) = Eulidean_Distance_Function(Training_Set(i),Test_Set(j));
    end
end

Accuracy_Vector = zeros(Samples_Count-1, 1);
Training_Labels = transpose(Training_Labels);
Predicted_Labels_Matrix = zeros((Test_Samples_Count), 1);


y_pred = zeros(Test_Samples_Count, 1);

for x =1: Test_Samples_Count
    Temp_Vec = [Distance_Matrix(:, x) Training_Labels];
    Temp_Vec = sortrows(Temp_Vec, 1);
    Distance_Matrix_Temp = Temp_Vec(1:k, :);
    W_Range = zeros(k, 1);
    Cancer_Counter = 0;
    Normal_Counter = 0;
    Dk = max(Distance_Matrix_Temp(:, 1));
    D1 = min(Distance_Matrix_Temp(:, 1));
    for i=1: k
        Di = Distance_Matrix_Temp(i, 1);
        if Dk == D1
            W_Range(i) = 1;
        else
            W_Range(i) = Weighted_Distance_function(Dk, Di, D1);
        end
        
        if Distance_Matrix_Temp(i, 2) == 1
            Normal_Counter = Normal_Counter + W_Range(i);
        else
            Cancer_Counter = Cancer_Counter + W_Range(i);
        end
    end
    if Normal_Counter > Cancer_Counter
        y_pred(x) = 1;
    else
        y_pred(x) = -1;
    end
end

end