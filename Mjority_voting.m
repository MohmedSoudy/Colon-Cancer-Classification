function [ Predicted_Labels ] = Mjority_voting( Bayes_Labels , mknn_Labels , SVM_Labels )
%Assign the final label based on the majority voting between 3 classifiers 
[N_of_Samples , N_of_Coloumns] = size(Bayes_Labels);
for i = 1 :N_of_Samples
    if (SVM_Labels(i) == mknn_Labels(i))
        Predicted_Labels(i) = SVM_Labels(i);
    elseif (mknn_Labels(i) == Bayes_Labels(i))
        Predicted_Labels(i) = mknn_Labels(i);
    elseif (SVM_Labels(i) == Bayes_Labels(i))
        Predicted_Labels(i) = SVM_Labels(i);
    end
end
end

