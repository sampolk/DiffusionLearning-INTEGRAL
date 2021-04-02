%%

[coeff,score,latent,tsquared,explained] = pca(X, 'NumComponents', 1);

A = mat2gray(reshape(score,500,500));
[L,N] = superpixels(A,10000);
L = reshape(L, length(X),1); 

X_small = zeros(N,372);
for i = 1:N
    
    X_small(i,:) = mean(X(L == i,:),1);
    
end
    
