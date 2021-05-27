function x = ridge_regression(M,y, num_lambdas)

lambdas = 10.^(-1:num_lambdas);

loss = zeros(size(lambdas));

parfor i = 1:num_lambdas
    x = (cov(M) + lambdas(i)*eye(size(M,2)))\(M')*y;
    loss(i) = norm(M*x - y)^2 + lambdas(i)*(norm(x)^2);    
end

[~,i] = min(loss);
x = (cov(M) + lambdas(i)*eye(size(M,2)))\(M')*y;
