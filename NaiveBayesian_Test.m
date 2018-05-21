function Predicted_Labels = NaiveBayesian_Test(Mus, Sigmas, Testing_Set, Test_Number, N_of_Features_Cancer)
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