function x = ridge_regression(M,y, num_lambdas)
%{
Purpose: Finds the x that minimizes 

             L(x) = ||Mx-y||_2^2 + lambda ||x||_2^2

         where lambda ranges 10.^(-1:num_lambdas). 

        We output the x that minimizes L(x) across lambda choices evaluated

Inputs: M:              Dxc matrix. 
        y:              Dx1 vector
        num_lambdas:    Minimum exponent for lambda choices.       
%}

lambdas = 10.^(-1:num_lambdas);

loss = zeros(size(lambdas));

parfor i = 1:num_lambdas
    x = (cov(M) + lambdas(i)*eye(size(M,2)))\(M')*y;
    loss(i) = norm(M*x - y)^2 + lambdas(i)*(norm(x)^2);    
end

[~,i] = min(loss);
x = (cov(M) + lambdas(i)*eye(size(M,2)))\(M')*y;
