function [Feature_Matrix]  = Read_data(Path)
% Read Data from File containing Matrix
File = fopen(Path);
Line = fgets(File);
i = 1;
while ischar(Line) 
    Split = strsplit(Line);
    %Label_vector(i) = str2double(Split{1,1});
    i = i + 1;
    S = size(Split);
    for j = 1 : S(2)
        Temp = strsplit(Split{j},':');
        Feature_Matrix(i-1  , j) = str2double(Temp(2));
    end
    Line = fgets(File);        
end
dlmwrite('ColonCancer_Features.txt',Feature_Matrix);
