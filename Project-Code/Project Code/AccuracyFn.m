function Accuracy = AccuracyFn(Confusion_Matrix, Test_Number)

%CALCULATING ACCURACY
Diagonal_Sum = Confusion_Matrix(1, 1) + Confusion_Matrix(2, 2);
Accuracy = (Diagonal_Sum / Test_Number) * 100;

%WRITE ACCURACY TO FILE
Fileid = fopen('C:\Users\Dell\Desktop\Accuracy.txt','a+');
S2 = 'Accuracy = ';
fprintf(Fileid,'%s %s',S2,num2str(Accuracy), '%');
fclose(Fileid); 
%winopen('C:\Users\Dell\Desktop\Accuracy.txt');
end