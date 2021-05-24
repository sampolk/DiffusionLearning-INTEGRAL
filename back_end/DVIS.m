function Clusterings_final = DVIS(X, Hyperparameters, Dist_NN, Idx_NN, Clusterings_init, Y)

if nargin == 4
    % Extract initial Clustering
    density =  KDE_large(Dist_NN, Hyperparameters);
    G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    Clusterings_init = MLUND_large(X, Hyperparameters, G, density);
    
    % Use unsupervised estimate for our initial clustering guess
    C = Clusterings_init.Labels(:,Clusterings_init.TotalVI.Minimizer_Idx);
    K = Clusterings_init.K(Clusterings_init.TotalVI.Minimizer_Idx);
    t = Clusterings_init.TimeSamples(Clusterings_init.TotalVI.Minimizer_Idx);

elseif nargin == 5
    G = Clusterings_init.Graph;
    density = Clusterings_init.Density;
    
    % Use unsupervised estimate for our initial clustering guess
    C = Clusterings_init.Labels(:,Clusterings_init.TotalVI.Minimizer_Idx);
    K = Clusterings_init.K(Clusterings_init.TotalVI.Minimizer_Idx);
    t = Clusterings_init.TimeSamples(Clusterings_init.TotalVI.Minimizer_Idx);

else
    

    G = Clusterings_init.Graph;
    density = Clusterings_init.Density;
    
    % Use the best clustering extracted by M-LUND as our initial clustering guess
    [~,t] = measure_performance(Clusterings_init,Y);  
    
    C = Clusterings_init.Labels(:,t);
    K = Clusterings_init.K(t);
    t = Clusterings_init.TimeSamples(t);

end

%% 


%% 
n_eigs = find(G.EigenVals(1:10).^t>Hyperparameters.Tolerance, 1,'last');
DiffusionMap = zeros(size(G.EigenVecs,1),n_eigs-1);

for k = 2:n_eigs
    DiffusionMap(:,k-1) = G.EigenVecs(:,k).*(G.EigenVals(k)^t);
end


%% Extract Dictionary
A = [];
for k = 1:K
    
    Xk = DiffusionMap(C==k, :);
    losses = zeros(size(DiffusionMap,2)-1,1);
    for c = 1:length(losses)
        try 
            [~, ~, ~, loss] = mvcnmf_shell(Xk, c+1);
            losses(c) = loss;
        catch
            % Didn't converge using Conjugate Gradient Descent. 
            % So, try standard gradient descent. 
            try 
                [~, ~, ~, loss] = mvcnmf_shell(Xk, c+1,2); 
                losses(c) = loss;
            catch
                losses(c) = NaN; % NMF Didn't converge for either algorithm.
            end
        end
    end
    [~,c] = min(losses);
    [A_est, ~, ~, ~] = mvcnmf_shell(Xk, c+1);
    A = [A; A_est];
end
M = MinSubset(A',Hyperparameters.Tolerance); 
A = hyperNnls(DiffusionMap', M)';
A = A./sum(A,2);
A(:, sum(A,1)==0) = []; % Get rid of empty columns

[Idx,Dist] = knnsearch(A,A,'K', Hyperparameters.DiffusionNN+1);
Idx = Idx(:,2:end);
Dist = Dist(:,2:end);

%%  Compute Graph on A and cluster A using it. 

G = extract_graph_large(A, Hyperparameters, Idx, Dist);
Clusterings_final = MLUND_large(A, Hyperparameters, G, density);


