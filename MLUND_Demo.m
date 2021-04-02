%{
Remarks:

 -  You must have run india_forest.m first and downloaded the file titled
    'normalized_data_NNs.mat'. Otherwise, this script will not work. 

 -  On my (Sam's) computer, I can only store weight matrices with 1000 
    nearest neighbors in my workspace due ot memory constraints. My 
    computer has 8 GB of RAM.

%}
%% Load data and nearest neighbors

load('normalized_data_NNs.mat')
Dist_NN = [D1, D2, D3, D4];
Idx_NN = [Idx1, Idx2, Idx3, Idx4];
clear 'D1' 'D2' 'D3' 'D4' 'Idx1' 'Idx2' 'Idx3' 'Idx4'

%% Set Hyperparameters
% 
Hyperparameters.DensityNN = 200; % must be ≤ 3500
Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN),50, 'all');
Hyperparameters.SpatialParams.ImageSize = [500,500];
Hyperparameters.DiffusionNN = 100;
Hyperparameters.NEigs = 10;
Hyperparameters.DiffusionTime = 0;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.Beta = 3;
Hyperparameters.Tau = 10^(-5);

% Endmember algorithm specifications
Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR';
Hyperparameters.EndmemberParams.K = 7; 

save_on = 0;
plot_on = 1;

%% Calculate Density

p = KDE_large(Dist_NN, Hyperparameters);

if save_on
    save('density.mat','p', 'Hyperparameters')
end
if plot_on
  
    imagesc(reshape(log10(p), M,N))
    colorbar
    xticks([])
    yticks([])
    title('$\log_{10}(p)$', 'interpreter' , 'latex')
    caxis([prctile(log10(p),1), prctile(log10(p),99)])
end

%% Calculate Graph Structure

G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

if save_on
    save(strcat('graph_adjesency-', num2str(Hyperparameters.DiffusionNN), '.mat'), 'G')
end

if plot_on 
    for i = 1:min([Hyperparameters.NEigs-1,12])
        subplot(3,4,i)
        imagesc(reshape(G.EigenVecs(:,1+i),500,500))
        pbaspect([1,1,1])
        title(strcat('Eigenvector-', num2str(i+1)))
        xticks([])
        yticks([])
        colorbar
    end
end

%% Calculate M-LUND Clusterings

Clusterings = MLUND_large(X, Hyperparameters, G, p);

if save_on
    save(strcat('MLUND.mat'), 'Clusterings')
end 

if plot_on 
    
    nontrivial_K = unique(Clusterings.K(and(Clusterings.K>1, Clusterings.K<n/2)));
    n_nontrivial_K = length(nontrivial_K);
    
    if n_nontrivial_K == 1
        n_row = 1;
        n_col = 1;
   
    elseif mod(n_nontrivial_K, 3) == 1 && mod(n_nontrivial_K, 4) == 1
        
        n_row = ceil(sqrt(n_nontrivial_K));
        n_col = ceil(sqrt(n_nontrivial_K));
        
    elseif mod(n_nontrivial_K, 4) == 1

        n_row = 3;
        n_col = ceil(n_nontrivial_K/3);

    elseif mod(n_nontrivial_K, 3) == 1

        n_row = 4;
        n_col = ceil(n_nontrivial_K/4);
        
    end
       
    for plot_idx = 1:n_nontrivial_K
        subplot(n_row, n_col, plot_idx)
        
        t = find(Clusterings.K == nontrivial_K(plot_idx), 1, 'first');
    
        imagesc(reshape(Clusterings.Labels(:,t),500,500))
        pbaspect([1,1,1])
        title(strcat('LUND Clustering, $K$=', num2str(Clusterings.K(t)), ', $t$=', num2str(Clusterings.TimeSamples(t))), 'interpreter', 'latex')
        xticks([])
        yticks([])
    end
end

%% Calculate M-LUND Clusterings with Endmembers

Clusterings = MLUNDEndmember(X, Hyperparameters, Idx_NN, Dist_NN, G);

if save_on
    save(strcat('MLUND_Endmember.mat'), 'Clusterings')
end

if plot_on 
    
    nontrivial_K = unique(Clusterings.K(and(Clusterings.K>1, Clusterings.K<n/2)));
    n_nontrivial_K = length(nontrivial_K);
    
    if n_nontrivial_K == 1
        n_row = 1;
        n_col = 1;
        
    elseif mod(n_nontrivial_K, 3) == 1 && mod(n_nontrivial_K, 4) == 1
        
        n_row = ceil(sqrt(n_nontrivial_K));
        n_col = ceil(sqrt(n_nontrivial_K));
        
    elseif mod(n_nontrivial_K, 4) == 1

        n_row = 3;
        n_col = ceil(n_nontrivial_K/3);

    elseif mod(n_nontrivial_K, 3) == 1

        n_row = 4;
        n_col = ceil(n_nontrivial_K/4);
        
    end
       
    for plot_idx = 1:n_nontrivial_K
        subplot(n_row, n_col, plot_idx)
        
        t = find(Clusterings.K == nontrivial_K(plot_idx), 1, 'first');
    
        imagesc(reshape(Clusterings.Labels(:,t),500,500))
        pbaspect([1,1,1])
        title(strcat('LUND Clustering, $K$=', num2str(Clusterings.K(t)), ', $t$=', num2str(Clusterings.TimeSamples(t))), 'interpreter', 'latex')
        xticks([])
        yticks([])
    end
end
