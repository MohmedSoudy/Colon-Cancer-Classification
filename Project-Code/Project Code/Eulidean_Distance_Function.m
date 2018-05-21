function Distance = Eulidean_Distance_Function(Training_Vector , Testing_Vector)
Distance = sqrt(sum((Training_Vector - Testing_Vector) .^ 2));
end