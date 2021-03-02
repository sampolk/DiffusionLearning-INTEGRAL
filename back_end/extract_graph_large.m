function G = extract_graph_large(X, Hyperparameters, Idx_NN)

n = length(X);
NN = Hyperparameters.DiffusionNN;
n_eigs = Hyperparameters.NEigs;

% Preallocate memory for sparse matrix calculation
ind_row = repmat((1:n)', 1,NN);  % row indices for nearest neighbors.
ind_col = Idx_NN(:,1:NN);         % column indices for nearest neighbors.
W = sparse(ind_row, ind_col, ones(n,NN)); % construct W
A = adjacency((graph((W+W')./2)));   % Convert W to adjesency matrix. (W+W)./2 forces symmetry.
Dinv = spdiags(1./sum(A)',0,n,n);    % D^{-1}.

[V,D, flag] = eigs(Dinv*A, n_eigs, 'largestabs'); % First n_eigs eigenpairs of transition matrix

if flag
    disp('Convergence Failed.')
    G = NaN;
else
    [lambda,idx] = sort(diag(abs(D)),'descend');
    lambda(1) = 1;
    V = real(V(:,idx));

    G.Hyperparameters = Hyperparameters;
    G.EigenVecs = V;
    G.EigenVals = lambda;
end