function Confusion_Matrix = Confusion_Matrix_Function(y_true, y_pred)
C = 2;
Confusion_Matrix = zeros(C, C);
[Test_Number , X] = size(y_pred);

for i = 1:Test_Number
    if((y_pred(i) == 1) && (y_true(i) == 1)) %Normal Right
        Confusion_Matrix(1, 1) = Confusion_Matrix(1, 1) + 1;
    elseif((y_pred(i) == -1) && (y_true(i) == 1))%Normal Wrong
        Confusion_Matrix(1, 2) = Confusion_Matrix(1, 2) + 1;
        
    elseif((y_pred(i) == -1) && (y_true(i) == -1)) %Cancer Right
        Confusion_Matrix(2, 2) = Confusion_Matrix(2, 2) + 1;
    else %Cancer Wrong
        Confusion_Matrix(2, 1) = Confusion_Matrix(2, 1) + 1;
    end
end