function [ Training_Set_Cancer , Training_Set_Normal  ,Testing_Set  ,Label_Vector , Testing_Label] = Data_Preprocessing( Feature_Matrix , Labels)
%Process Data For Training & Cross validation 
Training_Set_Normal = Feature_Matrix(1 : 12 , :);  
Training_Set_Cancer = Feature_Matrix(23 : 42 , :); 
Label_Vector = Labels(1 : 12 , :);
Label_Vector = [Label_Vector ;Labels(23 : 42, :)];
Testing_Set = Feature_Matrix(13 : 22 , :);
Testing_Set = [Testing_Set ; Feature_Matrix(43 : 62 , :)];
Testing_Label = Labels(13 :22 , :);
Testing_Label = [Testing_Label ; Labels(43 : 62 , :)];
end

