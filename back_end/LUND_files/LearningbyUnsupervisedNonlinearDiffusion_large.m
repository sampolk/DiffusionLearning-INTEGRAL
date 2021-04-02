function [C, K, Dt] = LearningbyUnsupervisedNonlinearDiffusion_large(X, Hyperparameters, t, G, p)
%{
 - This function produces a structure with multiscale clusterings produced
   with the LUND algorithm, presented in the following paper. 

        - Maggioni, M., J.M. Murphy. Learning by Unsupervised Nonlinear 
          Diffusion. Journal of Machine Learning Research, 20(160), 
          pp. 1-56. 2019.
    
   and analyzed further in the following papers:

        - Murphy, James M and Polk, Sam L., 2020. A Multiscale Environment 
          for Learning By Diffusion. arXiv preprint, arXiv:2102.00500.
        - Polk, Sam L. and Murphy James M., 2021. Multiscale Spectral-
          Spatial Diffusion Geometry for Hyperspectral Image Clusteri
          (In Review)

Inputs: X:                      Data matrix.
        Hyperparameters:        Optional structure with graph parameters
                                with required fields:  
        t:                      Diffusion time parameter.
        G:                      Graph structure computed using  
                                'extract_graph_large.m' 
        p:                      Kernel Density Estimator.
        time_idx:                      Kernel Density Estimator.

Output: 
            - C:                n x 1 vector storing the LUND clustering 
                                of X at time t.
            - K:                Scalar, number of clusters in C.
                                clusters in the Labels(:,t) clustering.
            - Dt:               n x 1 matrix storing \mathcal{D}_t(x). 

© 2021 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu
%}  

if ~isfield(Hyperparameters, 'NEigs')
    Hyperparameters.NEigs = size(G.EigenVecs,2);
end
if ~isfield(Hyperparameters, '')
    Hyperparameters.DtNNs = 100;
end

n = length(X);

% Calculate diffusion map
DiffusionMap = zeros(n,Hyperparameters.NEigs);
for l = 1:size(DiffusionMap,2)
    DiffusionMap(:,l) = G.EigenVecs(:,l).*(G.EigenVals(l).^t);
end

disp('Diffusion Map Calculated')

tic
% Compute Hyperparameters.NumDtNeighbors Dt-nearest neighbors.
[IdxNN, D] = knnsearch(DiffusionMap, DiffusionMap, 'K', Hyperparameters.NumDtNeighbors);
disp('Nearest Neighbor Searches Complete')
toc
tic
% compute rho_t(x), stored as rt
rt = zeros(n,1);
p_max = max(p);
for i=1:n
    if p(i) == p_max
        rt(i) = max(pdist2(DiffusionMap(i,:),DiffusionMap));
    else
        [~, IdxNN_dense, ~] = intersect(IdxNN(i,:), find(p>p(i)), 'stable');  
        
        if ~isempty(IdxNN_dense)
            % In this case, at least one of the Hyperparameters.NumDtNeighbors Dt-nearest neighbors
            % of X(i,:) is also higher density, so we have already
            % calculated rho_t(X(i,:)).
            dt_temp = D(i,IdxNN_dense);
            rt(i) = dt_temp(1);
        else
            % In this case, none of the first Hyperparameters.NumDtNeighbors Dt-nearest neighbors of
            % X(i,:) are also higher density. So, we do the full search.
            rt(i) = min(pdist2(DiffusionMap(i,:),DiffusionMap(p>p(i),:)));
        end
    end
    if mod(i,5000)==0
        disp(strcat('Rho_t(x) Calculation, ', num2str(floor(i/n*100)),  '% Complete'))
    end
end
disp('Rho_t(x) Calculation Complete')
toc
% Extract Dt(x) and sort in descending order
Dt = rt.*p;
[~, m_sorting] = sort(Dt,'descend');


% Determine K based on the ratio of sorted Dt(x_{m_k}). 
if isfield(Hyperparameters, 'K_Known')
    K = Hyperparameters.K_Known;
else
    [~, K] = max(Dt(m_sorting(1:n-1))./Dt(m_sorting(2:n)));
end
disp('Cluster Modes Calculated')

if K == 1
    C = ones(n,1);
else

    C = zeros(n,1);
    % Label modes
    C(m_sorting(1:K)) = 1:K;

    % Label non-modal points according to the label of their Dt-nearest
    % neighbor of higher density that is already labeled.
    [~,l_sorting] = sort(p,'descend');
    tic
    for j = 1:n
        i = l_sorting(j);
        if C(i)==0 % unlabeled point
            
            candidates = find(and(C>0, p>p(i))); % Labeled, higher-density points.
            IdxNN_i = intersect(IdxNN(i,:), candidates, 'stable');
            
            if isempty(IdxNN_i)
                % None of the Dt-nearest neighbors are also higher-density
                % & labeled. So, we do a full search.
                [~,temp_idx] = min(pdist2(DiffusionMap(i,:), DiffusionMap(candidates,:)));
                C(i) = C(candidates(temp_idx));    
            else
                % At least one of the Dt-nearest neighbors is higher
                % density & labeled. So, we pick the closest point.
                C(i) = C(IdxNN_i(1));                
            end
        end
        if mod(j,5000)==0
            disp(strcat('Non-modal labeling, ', num2str(floor(j/n*100)),  '% Complete'))
        end
    end
    
    toc

end 

disp(strcat('LUND Clustering Complete.'))
