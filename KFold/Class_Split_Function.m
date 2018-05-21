function [Normal_Set, Cancer_Set] = Class_Split_Function(Training_Set, Label_Vector)
[Size, Features] = size(Training_Set);
Normal_Set = zeros(0, Features);
Cancer_Set = zeros(0, Features);
for i=1: Size
    if Label_Vector(i) == 1
        Normal_Set = [Normal_Set; Training_Set(i, :)];
    else    
        Cancer_Set = [Cancer_Set; Training_Set(i, :)];
    end
end
end