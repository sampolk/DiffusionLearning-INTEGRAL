function [purity, endmembers, abundances] = compute_purity(X,Hyperparameters)
%{
Purpose: Calculates the endmember decomposition of a Hyperspectral image X
         as well as pixel purity. The specific endmember extraction method
         can be any of the following:

            - ATGP
            - N-FINDR
            - VCA
            - N-FINDR-I
            - VCA-I

Inputs:     X:                  (M*N)xD data matrix. 
            Hyperparameters:    Structure with the following fields:
                - SpatialParams:    Stores the dimensions of the original image. 
                - DiffusionTime:    Diffusion time parameter.
                - DiffusionNN:      Number of nearest neighbors in KNN graph.
                - EndmemberParams:  Endmember extraction algorithm and number of endmembers (Optional).
                - WeightType:       Equal to either 'adjesency' or 'gaussian' (Optional).
                - Sigma:            If WeightType == 'gaussian', then diffusion scale parameter Sigma>0 required.
                - IncludeDensity:   1 if density is to be included, 0 otherwise (Optional).
                - DensityNN:        If IncludeDensity == 1, then number of nearest neighbors to comput KDE is required.
                - Sigma0:           If IncludeDensity == 1, then KDE bandwidth Sigma0>0 is required.

Outputs:    
            endmembers:     Calculated endmembers.
            purity:         Purity of pixels, as measured by endmember unmixing.
%}


if ~isfield(Hyperparameters, 'EndmemberParams')
    Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR';
    Hyperparameters.EndmemberParams.K = 7;
end

% Size of HSI.
M = Hyperparameters.SpatialParams.ImageSize(1); 
N = Hyperparameters.SpatialParams.ImageSize(2);

% Number of endmembers
K = Hyperparameters.EndmemberParams.K; 

if strcmp(Hyperparameters.EndmemberParams.Algorithm, 'ATGP')
    
    [endmembers,~] = EIA_ATGP(X', K);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'N-FINDR')
    
    [endmembers,~] = EIA_NFINDR(X',K,200, Hyperparameters.SpatialParams.ImageSize);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'N-FINDR-I')

    [endmembers,~] = EIA_NFINDR(X',K+2,200, Hyperparameters.SpatialParams.ImageSize); % 2 additional endmembers. Truncated later.
    
    endmembers = endmembers(:,3:K+2);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'VCA')

    [ endmembers, ~, ~ ] = hyperVca(double( X'), K );
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'VCA-I')
    
    [ endmembers, ~, ~ ] = hyperVca( double(X'), K+2 ); % 2 additional endmembers. Truncated later.
    
    endmembers = endmembers(:,3:K+2);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'GRNMF')
    
    [endmembers, abundances] = GroupRobustNMF(X, K, 1e-4);
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'PLM')
    
    Unmixing = PerturbedLinearMixing(X,[M,N], K);
    
    abundances = Unmixing.Abundance;
    endmembers = Unmixing.Endmembers;
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'MVC-NMF')
    [endmembers, abundances, ~, ~] = mvcnmf_shell(X, K);
end

abundances = reshape(abundances,M*N,K);
abundances = abundances./sum(abundances,2); % Row normalize to make a distribution
purity = max(abundances,[],2);

    
end 
