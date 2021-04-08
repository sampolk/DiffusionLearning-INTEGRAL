function W = spatial_weight_matrix(X,Hyperparameters)
%{
 - This function computes the weight matrix for a modified graph that 
   directly incorporates spatial geometry into graph diffusion. See paper 
   below for more.

        - Polk, Sam L. and Murphy James M., 2021. Multiscale Spectral-
          Spatial Diffusion Geometry for Hyperspectral Image Clustering. 
          (In Review)

Inputs: X:                      (M*N)xD Data matrix .
        Hyperparameters:        Structure with graph parameters with the 
                                required fields: 
            - SpatialParams:    Stores the dimensions of the original image
            - DiffusionNN:      Number of nearest neighbors in KNN graph.
            - WeightType:       Equal to either 'adjesency' or 'gaussian' (Optional).
            - Sigma:            If WeightType == 'gaussian', then diffusion scale parameter Sigma>0 required.

Output: W:                  Weight matrix for graph.

© 2021 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu
%}


% If spatial information is included in hyperparameters structure, we
% incorporate that into the diffusion process.
R = Hyperparameters.SpatialParams.SpatialRadius;
M = Hyperparameters.SpatialParams.ImageSize(1); 
N = Hyperparameters.SpatialParams.ImageSize(2);
n = M*N; % Number of pixels in X
NN = Hyperparameters.DiffusionNN;

if strcmp(Hyperparameters.WeightType, 'gaussian')
    sigma = Hyperparameters.Sigma;
end

indices = 1:n;

idx_row = zeros(n, NN);  % Rows where nonzero elements appear in W
idx_col = zeros(n, NN);  % Columns where nonzero elements appear in W
vals    = zeros(n, NN);  % Variable to store edge weights


times = zeros(n, 3);


for idx = 1:n
    
    tic
    [i,j] = ind2sub([M,N], idx);
    NN_Idx=FindNeighbors([i,j], R, M, N); % indices of the spatial nearest neighbors of ij pixel of X.
    NN_Count=length(NN_Idx);
    times(idx,1) = toc;
    
    tic
    Xi_rep = repmat(X(idx,:),NN_Count,1); % Spectra of ij pixel of X, repeated in each row.
    X_NNs = X(NN_Idx,:); % Spectra of spatial nearest neighbors of ij pixel
        
    DistTemp=sqrt(sum((Xi_rep - X_NNs).^2,2)); % distance between ij pixel of X and its spatial nearest neighbors
    [NN_Dists,Idx]=sort(DistTemp,'ascend');
    NN_Idx=NN_Idx(Idx);
    
    times(idx,2) = toc;
    
    
    tic
    if NN_Count>=NN
    
        % Distances between X(idx,:) and nearest neighbors of X(idx,:) in spatial search
        idx_row(idx,:) = idx;
        idx_col(idx,:) = NN_Idx(1:NN);

        if strcmp(Hyperparameters.WeightType, 'adjesency')
            vals(idx,:) = ones(NN,1);
        else
            vals(idx,:) = exp(-(NN_Dists(1:NN).^2)./(sigma^2));
        end  
        
    else
        
        nmissing = NN-NN_Count;
        
        temp = indices(~(ismember(indices,NN_Idx)));
        col_filler = temp(1:nmissing);
        val_filler = zeros(1,nmissing);
        
        % Distances between X(idx,:) and nearest neighbors of X(idx,:) in spatial search
        idx_row(idx,:) = idx;
        idx_col(idx,:) = [NN_Idx, col_filler];

        if strcmp(Hyperparameters.WeightType, 'adjesency')
            vals(idx,:) = [ones(NN_Count,1),val_filler] ;
        else
            vals(idx,:) = [exp(-(NN_Dists.^2)./(sigma^2)),val_filler];
        end          
        
    end    
    times(idx,3) = toc;
    
    disp(idx/n)
end

W = sparse(idx_row, idx_col, vals);

end

