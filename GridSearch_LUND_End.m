%{
Remarks:
 -  You must have run india_forest.m first and downloaded the file titled
    'normalized_data_NNs.mat'. Otherwise, this script will not work. 
 -  On my (Sam's) computer, I can only store weight matrices with 1000 
    nearest neighbors in my workspace due ot memory constraints. My 
    computer has 8 GB of RAM.
%}

load('normalized_data_NNs.mat')
Dist_NN = [D1, D2, D3, D4];
Idx_NN = [Idx1, Idx2, Idx3, Idx4];
clear 'D1' 'D2' 'D3' 'D4' 'Idx1' 'Idx2' 'Idx3' 'Idx4'


%% Set Hyperparameters

% Hyperparameters.DensityNN = 500; % must be ≤ 3500
% Hyperparameters.DiffusionNN = 100;
% Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN),50, 'all');
% Hyperparameters.Sigma = prctile(Dist_NN(:,1:Hyperparameters.DensityNN),50, 'all');


% Hyperparameters.WeightType = 'gaussian';
% Hyperparameters.NumClusterBound = 10000;

% Hyperparameters.K_Known = 5; 
% Optional parameter. If included, it is assigned as the number of clusters. Else, K is found fully unsupervised.

Hyperparameters.SpatialParams.ImageSize = [500,500];
Hyperparameters.NEigs = 10;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.DiffusionTime = 512;

% Endmember algorithm specifications
Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR';
Hyperparameters.EndmemberParams.K = 7; 

save_on = 0;
plot_on = 1;%% Set Hyperparameters

%% Search across grid of clusterings
% Before using, go to an empty folder, where results will be saved. 

NNs = [100, 200, 500, 1000, 1500, 2000];
prctiles = 1:14:99;

for i = 1:length(NNs)
    
    Hyperparameters.DiffusionNN = NNs(i);
    Hyperparameters.DensityNN = NNs(i); % must be ≤ 3500
    
    for j = 1:length(prctiles)
        Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN),prctiles(j), 'all');
    
    
        % Calculate Density

        p = KDE_large(Dist_NN, Hyperparameters);
        p(p<prctile(p,0.01)) = prctile(p,0.01); % Density is rapidly decaying for low-density points, so we set a floor at the 99.99th percentile of p to aid in mode detection. 


        G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

        % Clusterings = MLUND_large(X, Hyperparameters, G, p);
        [C, K, Dt] = LearningbyUnsupervisedNonlinearDiffusion_large(X, Hyperparameters, Hyperparameters.DiffusionTime, G, p);
        Cluster = LUNDEndmember(X, Hyperparameters, Idx_NN, Dist_NN, G);
        C1 = Cluster.Labels;
        Clusterings.C = C;
        Clusterings.C1 = C1;
        

        save(strcat('LUND-', num2str(NNs(i)), '-', num2str(prctiles(j))), 'Clusterings')
        disp([i,j]/[length(NNs), 24])

    end
    
end

%% Visualize Results
% Saves a visualization of all clusterings learned by M-LUND in a folder.

% To use, follow the following procedure:
%   1. Move to the folder where your clusterings are stored. 
%   2. Review images that are saved to visualize different clusterings. 

% fig = figure('units','normalized','outerposition',[0 0 1 1]);

files = dir('*.mat');
n_files = length(files);

% figure
for i = 1:n_files
    file_name = files(i).name;
    load(file_name);
    
    idces = [1];
    
    n_plots = length(idces);
    
    f = figure;
    imagesc(reshape(Clusterings.C,500,500))
    pbaspect([1,1,1])
    title(strcat('LUND Clustering, $K$=', num2str(K), ', $t$=', num2str(512)), 'interpreter', 'latex')
    xticks([])
    yticks([])
    colorbar
    
    saveas(f, strcat(file_name(1:end-3), 'jpeg'))
    
    f1 = figure;
    imagesc(reshape(Clusterings.C1,500,500))
    pbaspect([1,1,1])
    title(strcat('LUND End Clustering, $K$=', num2str(K), ', $t$=', num2str(512)), 'interpreter', 'latex')
    xticks([])
    yticks([])
    colorbar
    
    saveas(f1, strcat('END-',file_name(1:end-3), 'jpeg'))
end
close all
