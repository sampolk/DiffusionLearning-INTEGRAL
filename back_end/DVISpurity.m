function purity = DVISpurity(G, tolerance, C)
%{
This script does the following: 

    1.  For each cluster in C, we calculate a dictionary in the coordinates
        of that cluster in the coordinates of the diffusion map.
    2.  We then implement non-negative constrained least squares to get 
        abundance estimates, which are row-normalized. 
    3.  We then take the maximum abundance for each point as our purity
        estiamte.

Inputs: G:          Graph structure computed using extract_graph_large.m.
        tolerance:  We get rid of redundant dictionary atoms by discarding 
                    tolerance tau of each other. 
        C:          Initial Clustering guess.

Output: purity:     Our estimate for pixel purity.

%}


K = length(unique(C));

% ---------------------- Extract Dictionary Elements ----------------------
% First, we calculate the Diffusion Map at time t. 
 
t = ceil(2.^(-log2(1-G.EigenVals(K+1))));
DiffusionMap = zeros(size(G.EigenVecs,1),2);
parfor k = 2:K
    DiffusionMap(:,k-1) = G.EigenVecs(:,k).*(G.EigenVals(k)^t);
end


% For each cluster from our initial clustering, we implement MVC-NMF across
% a grid of choices for the number of latent dictionary elements, c
M = []; 
for k = 1:K
        
    cluster = DiffusionMap(C==k, :); 
    losses = zeros(size(DiffusionMap,2)-1,1);    
    if size(cluster,1) == 1 % singleton cluster, so don't run MVC-NMF.
        losses(:) = NaN;
    else
        % We implement all choices of c and minimize loss function. 
        parfor c = 1:length(losses)
            try 
                [~, ~, ~, losses(c)] = mvcnmf_shell(cluster, c+1);
            catch
                
                losses(c) = NaN; % MVC-NMF Didn't converge. So, reject this choice of c
            end
        end
    end
    if sum(~isnan(losses))>0
        [~,c] = min(losses); % Best choice of c. 
        M = [M; mvcnmf_shell(cluster, c+1)];

    else
        % In this case, MVC-NMF didn't converge for any choice of c. 
        % We take the mean of the cluster in coordinates of DiffusionMap
        M = [M; mean(cluster,1)];
    end
end

% --------------------- Post-Processing of Dictionary ---------------------
 
% Get rid of redundant dictionary atoms
M = MinSubset(M', tolerance); 

% Implement non-negative constrained least squares to get abundance estimates
A = hyperNnls(DiffusionMap', M)';

% Ensure that no zero entries exist
idx = find(sum(A,2) == 0);

% If there are any, we perform a ridge regression to get the coefficients.
temp = zeros(length(idx),size(A,2));
parfor j = 1:length(idx)
    temp(j,:) = ridge_regression(M,DiffusionMap(idx(j),:)', 10);
end
A(idx,:) = temp;

% Row-normalize to make each row a probability distribution
A = A./sum(A,2); 

% Get rid of atoms with zero-mass.
A(:, sum(A,1)==0) = []; 

purity = max(A,[],2);