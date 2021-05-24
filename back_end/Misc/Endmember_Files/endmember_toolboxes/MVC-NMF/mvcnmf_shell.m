function [A_est, S_est, volume, loss] = mvcnmf_shell(data, c, alg_type)

if nargin == 2
    alg_type = 1;
end

data = data';
[~,n] = size(data);

% remove noise
[UU, ~, ~] = svds(data,c);
Lowmixed = UU'*data;
data = UU*Lowmixed;

% vca algorithm
[A_init, ~] = vca(data,'Endmembers', c,'verbose','off');

% FCLS
warning off;
AA = [1e-5*A_init;ones(1,length(A_init(1,:)))];
S_init = zeros(length(A_init(1,:)),n);
for j=1:n
    r = [1e-5*data(:,j); 1];
    S_init(:,j) = lsqnonneg(AA,r);
end

% PCA
[~,PrinComp] = pca(data);
meanData = mean(data,2)';

% test mvcnmf
tol = 1e-6;
maxiter = 150;
T = 0.015;

% use conjugate gradient to find A can speed up the learning
[A_est, S_est, volume, loss] = mvcnmf(data,A_init,S_init,PrinComp,meanData,T,tol,maxiter,2,alg_type);

A_est = A_est';
S_est = S_est';