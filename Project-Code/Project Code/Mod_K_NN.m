function [K, Accuracy, Conf_Matrix] = Mod_K_NN(Training_Set, Test_Set, Training_Labels, Test_Labels)
[Samples_Count, Features_Count] = size(Training_Set);
[Test_Samples_Count, Test_Features_Count] = size(Test_Set);
Distance_Matrix = zeros(Samples_Count, Test_Samples_Count);

for i = 1 : Samples_Count
    for j = 1 : Test_Samples_Count
     Distance_Matrix(i, j) = Eulidean_Distance_Function(Training_Set(i),Test_Set(j));
    end
end

Accuracy_Vector = zeros(Samples_Count-1, 1);
K_Vector = zeros(Samples_Count-1, 1);
Training_Labels = transpose(Training_Labels);
Predicted_Labels_Matrix = zeros(1, (Test_Samples_Count));
Test_Labels = transpose(Test_Labels);

for k =1: (Samples_Count) - 1
    K_Vector(k) = k;
    Range = k;
    y_pred = zeros(1, Test_Samples_Count);
    for x =1: Test_Samples_Count
        Temp_Vec = [Distance_Matrix(:, x) Training_Labels];
        Temp_Vec = sortrows(Temp_Vec, 1);
        Distance_Matrix_Temp = [Temp_Vec(1:Range, 1) Training_Labels(1:Range)];
        D_Range =  Distance_Matrix_Temp(1:Range, :);
        W_Range = zeros(Range);
        Cancer_Counter = 0;
        Normal_Counter = 0;
        for i=1: Range
            Dk = max(D_Range(:, 1));
            D1 = min(D_Range(:, 1));
            Di = D_Range(i, 1);
            if Dk == D1
                W_Range(i) = 1;
            else
                W_Range(i) = Weighted_Distance_function(Dk, Di, D1);
            end
            if D_Range(i, 2) == 1 
                Normal_Counter = Normal_Counter + W_Range(i);
            else
                Cancer_Counter = Cancer_Counter + W_Range(i, 1);
            end
        end
        if Normal_Counter > Cancer_Counter
            y_pred(x) = 1;
        else
            y_pred(x) = -1;
        end
    end
    Confusion_Matrix = Confusion_Matrix_Function(Test_Labels, y_pred);
    if k == 1
        Predicted_Labels_Matrix = y_pred;
    else
        Predicted_Labels_Matrix = [Predicted_Labels_Matrix; y_pred];
    end
    Accuracy_Vector(k) = AccuracyFn(Confusion_Matrix, Test_Samples_Count);
end

[val, idx] = max(Accuracy_Vector);
Conf_Matrix = Confusion_Matrix_Function(Test_Labels, Predicted_Labels_Matrix(idx, :));
Accuracy = val;
K = K_Vector(idx);
end




