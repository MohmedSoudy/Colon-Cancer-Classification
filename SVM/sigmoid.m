function G = sigmoid(U,V)
% Sigmoid kernel function with slope gamma and intercept c
gamma = 1;
c = 0.1;
G = tanh(gamma*U*V' + c);
end