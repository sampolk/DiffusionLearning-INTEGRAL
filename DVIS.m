function C = DVIS(G, p, Hyperparameters) 

K = Hyperparameters.K_Known;

% Spectral Clustering as Initialization
C = kmeans(G.EigenVecs(:,1:Hyperparameters.K_Known),K);

% ---------------------- Extract Dictionary Elements ----------------------
% First, we calculate the Diffusion Map at time t. 
t = floor(2.^(-log2(1-G.EigenVals(K+1))));
DiffusionMap = zeros(size(G.EigenVecs,1),2);
for k = 2:K
    DiffusionMap(:,k-1) = G.EigenVecs(:,k).*(G.EigenVals(k)^t);
end

% For each cluster from our initial clustering, we implement MVC-NMF across
% a grid of choices for the number of latent dictionary elements, c.

M = []; 
for k = 1:K
        
    cluster = DiffusionMap(C==k, :); 
    losses = zeros(size(DiffusionMap,2)-1,1);
    parfor c = 1:length(losses)
        try 
            [~, ~, ~, losses(c)] = mvcnmf_shell(cluster, c+1);
        catch
            % MVC-NMF Didn't converge using Conjugate Gradient Descent. 
            % So, try standard gradient descent. 
            try 
                [~, ~, ~, losses(c)] = mvcnmf_shell(cluster, c+1,2); 
            catch
                losses(c) = NaN; % MVC-NMF Didn't converge for either algorithm. So, reject this choice of c
            end
        end
    end
    [~,c] = min(losses); % Best choice of c. 
    M = [M; mvcnmf_shell(cluster, c+1)];
end

% --------------------- Post-Processing of Dictionary ---------------------

% Get rid of redundant dictionary atoms
M = MinSubset(M', Hyperparameters.Tolerance); 

% Implement non-negative constrained least squares to get abundance estimates
A = hyperNnls(DiffusionMap', M)';

% Row-normalize to make each row a probability distribution
A = A./sum(A,2); 

% Get rid of atoms with zero-mass.
A(:, sum(A,1)==0) = []; 

% ------------- Update Clustering Using Dictionary Abundances -------------

% Extract New Graph Using Abundances
[Idx, Dist] = knnsearch(A,A, 'K', Hyperparameters.AbundanceDiffusionNN+1);
Idx = Idx(:,2:end);  
Dist = Dist(:,2:end);

Hyperparameters.DiffusionNN = Hyperparameters.AbundanceDiffusionNN;
Hyperparameters.Sigma = Hyperparameters.AbundanceSigma;

GAb = extract_graph_large(A, Hyperparameters, Idx, Dist);

% Cluster using graph generated from abundances, A
C = LearningbyUnsupervisedNonlinearDiffusion_large(A, Hyperparameters, Hyperparameters.AbundanceTimeStep, GAb, p);

