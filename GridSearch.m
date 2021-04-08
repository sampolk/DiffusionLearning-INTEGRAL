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
% 
Hyperparameters.DensityNN = 200; % must be ≤ 3500
Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN),50, 'all');
Hyperparameters.SpatialParams.ImageSize = [500,500];
% Hyperparameters.DiffusionNN = 100;
Hyperparameters.NEigs = 10;
Hyperparameters.DiffusionTime = 0;
Hyperparameters.NumDtNeighbors = 200;

% Endmember algorithm specifications
Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR';
Hyperparameters.EndmemberParams.K = 5; 

save_on = 0;
plot_on = 1;%% Set Hyperparameters

%% Calculate Density

p = KDE_large(Dist_NN, Hyperparameters);
p(p<prctile(p,0.01)) = prctile(p,0.01); % Density is rapidly decaying for low-density points, so we set a floor at the 99.99th percentile of p to aid in mode detection. 

%% Search across grid of clusterings
% Before using, go to an empty folder, where results will be saved. 

NNs = [10:10:1000];


for i = 1:length(NNs)
    
    Hyperparameters.DiffusionNN = NNs(i);
    G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    
    Clusterings = MLUND_large(X, Hyperparameters, G, p);
    
    save(strcat('M-LUND-', num2str(NNs(i))), 'Clusterings')
    
    disp(i/length(NNs))
end

%% Visualize Results
% Saves a visualization of all clusterings learned by M-LUND in a folder.

% To use, follow the following procedure:
%   1. Move to the folder where your clusterings are stored. 
%   2. Review images that are saved to visualize different clusterings. 

fig = figure('units','normalized','outerposition',[0 0 1 1]);

files = dir('*.mat');
n_files = length(files);

% figure
for i = 1:n_files
    file_name = files(i).name;
    load(file_name);
    
    idces = [1];
    n_t = length(Clusterings.K);
    for t = 2:n_t
        
        if ~(Clusterings.K(t) == Clusterings.K(t-1)) % True if no. clusters changes at time t.
            idces = [idces,t];
        end
    end
    
    n_plots = length(idces);
    
    for j = 1:n_plots
        
        subplot(1,n_plots, j)
        eda(Clusterings.Labels(:,idces(j)), 0);
        title(strcat('LUND Clustering, $\log_2(t)=', num2str(log2(Clusterings.TimeSamples(idces(j)))), '$, $K_t=', num2str(Clusterings.K(idces(j))), '$'), 'interpreter', 'latex')
        colorbar off
    end
    saveas(fig, strcat(file_name(1:end-3), 'jpeg'))    
end
close all