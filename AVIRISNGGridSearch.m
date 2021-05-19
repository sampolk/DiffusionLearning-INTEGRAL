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

Hyperparameters.SpatialParams.ImageSize = [500,500];
Hyperparameters.DiffusionNN = 100;
Hyperparameters.NEigs = 10;
Hyperparameters.DiffusionTime = 0;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.Beta = 2;
Hyperparameters.Tau = 10^(-5);

% Endmember algorithm specifications
Hyperparameters.EndmemberParams.Algorithm = 'VCA-I';

% Use HySime to get Optimal Number of Endmembers
[kf,Ek]=hysime(reshape(HSI,M*N,size(HSI,3))'); 
Hyperparameters.EndmemberParams.K = kf;

save_on = 0;
plot_on = 1;%% Set Hyperparameters

%% Search across grid of clusterings
% Before using, go to an empty folder, where results will be saved. 

NNs = unique(round(floor(10.^(1.7:0.1:3)),-1));
prctiles = 5:5:95;

%% 
Hyperparameters.EndmemberParams.Algorithm = 'VCA-I';
[PixelPurity, endmembers] = compute_purity(X,Hyperparameters);

for i = 1:length(NNs)
    
    Hyperparameters.DiffusionNN = NNs(i);
    Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
        
    G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    
    % Compute Clusterings Using Solely Pixel Purity
    Hyperparameters.IncludeDensity = 0;
    Clusterings_VCA_I = MLUNDEndmember(X, Hyperparameters, Idx_NN, Dist_NN, G, PixelPurity, endmembers);
    
    for j = 1:length(prctiles)
        
        Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN),prctiles(j), 'all');

        % M-LUND In Its Original Form        
        Clusterings_KDE = MLUND_large(X, Hyperparameters, G, KDE_large(Dist_NN, Hyperparameters));

        % Compute Clusterings Using VCA-I Endmember Purity
        Hyperparameters.IncludeDensity = 1;
        Clusterings_VCA_I_KDE = MLUNDEndmember(X, Hyperparameters, Idx_NN, Dist_NN, G, PixelPurity, endmembers);

        save(strcat('results_', num2str(NNs(i)), '-',num2str(prctiles(j)),'.mat'), 'Clusterings_KDE', 'Clusterings_VCA_I', 'Clusterings_VCA_I_KDE')
        
        disp([i/length(NNs),j/length(prctiles)])

    end
    
end

save('summary_statistics','NNs', 'prctiles')
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
        elseif t== Clusterings.TotalVI.Minimizer_Idx % True if optimal clustering
            idces = [idces,t];
        end
    end
    
    n_plots = length(idces);
    
    for j = 1:n_plots
        
        subplot(1,n_plots, j)
        eda(Clusterings.Labels(:,idces(j)), 0);
        
        if idces(j)== Clusterings.TotalVI.Minimizer_Idx
            title({strcat('LUND Clustering, $\log_2(t)=', num2str(log2(Clusterings.TimeSamples(idces(j)))), '$, $K_t=', num2str(Clusterings.K(idces(j))), '$'), 'Total VI Minimizer'}, 'interpreter', 'latex')
        else
            title(strcat('LUND Clustering, $\log_2(t)=', num2str(log2(Clusterings.TimeSamples(idces(j)))), '$, $K_t=', num2str(Clusterings.K(idces(j))), '$'), 'interpreter', 'latex')
        end
        colorbar off
    end
    saveas(fig, strcat(file_name(1:end-3), 'jpeg'))    
end
close all


%% Visualize Results with Endmembers
% Saves a visualization of all clusterings learned by M-LUND in a folder.

% To use, follow the following procedure:
%   1. Move to the folder where your clusterings are stored. 
%   2. Review images that are saved to visualize different clusterings. 

fig = figure('units','normalized','outerposition',[0 0 1 1]);

files = dir('*.mat');
n_files = length(files);

for i = 1:n_files
    file_name = files(i).name;
    load(file_name);
    
    
% =============================== M-LUND ===============================
     
    n_T = length(Clusterings_KDE.K);
    ts = [1];
    for j = 2:n_T
        if Clusterings_KDE.K(j-1) ~= Clusterings_KDE.K(j) && Clusterings_KDE.K(j)<100
            ts = [ts,j];
        end
    end
    
    n_plots = length(ts);
    
    if n_plots<=4
        
        n_rows = 1; 
        n_cols = n_plots;
    elseif n_plots == 5
        n_rows = 2;
        n_cols = 3;
    else
        n_cols = 4;
        n_rows = ceil(n_plots/4);
    end
 
    
    for j = 1:n_plots
        
        t = ts(j);
        
        
        subplot(n_rows, n_cols, j)
        eda(Clusterings_KDE.Labels(:,t))
        title({'M-LUND Clustering'; strcat('$K=', num2str(Clusterings_VCA_I_KDE.K(t)), '$, $\log_2(t)=', num2str(log2(Clusterings_VCA_I_KDE.TimeSamples(t))), '$, ')}, 'interpreter', 'latex')
        colorbar off
        set(gca,'FontSize', 16)

    
    end
   
    saveas(fig, strcat(file_name(1:end-4), '-M-LUND'), 'jpeg')    
    
% ============================ M-LUND+VCA-I ===============================
     
    n_T = length(Clusterings_VCA_I.K);
    ts = [1];
    for j = 2:n_T
        if Clusterings_VCA_I.K(j-1) ~= Clusterings_VCA_I.K(j) && Clusterings_VCA_I.K(j)<100
            ts = [ts,j];
        end
    end
    
    n_plots = length(ts);
    
    if n_plots<=4
        
        n_rows = 1; 
        n_cols = n_plots;
    elseif n_plots == 5
        n_rows = 2;
        n_cols = 3;
    else
        n_cols = 4;
        n_rows = ceil(n_plots/4);
    end
 
    
    for j = 1:n_plots
        
        t = ts(j);
        
        
        subplot(n_rows, n_cols, j)
        eda(Clusterings_VCA_I.Labels(:,t))
        title({'M-LUND+VCA-I Clustering'; strcat('$K=', num2str(Clusterings_VCA_I_KDE.K(t)), '$, $\log_2(t)=', num2str(log2(Clusterings_VCA_I_KDE.TimeSamples(t))), '$, ')}, 'interpreter', 'latex')
        colorbar off
        set(gca,'FontSize', 16)
    
    end
   
    saveas(fig, strcat(file_name(1:end-4), '-M-LUND+VCA-I'), 'jpeg')    
    
% ========================== M-LUND+VCA-I+KDE =============================
     
    n_T = length(Clusterings_VCA_I_KDE.K);
    ts = [1];
    for j = 2:n_T
        if Clusterings_VCA_I_KDE.K(j-1) ~= Clusterings_VCA_I_KDE.K(j) && Clusterings_VCA_I_KDE.K(j)<100
            ts = [ts,j];
        end
    end
    
    n_plots = length(ts);
    
    if n_plots<=4
        
        n_rows = 1; 
        n_cols = n_plots;
    elseif n_plots == 5
        n_rows = 2;
        n_cols = 3;
    else
        n_cols = 4;
        n_rows = ceil(n_plots/4);
    end
 
    
    for j = 1:n_plots
        
        t = ts(j);
        
        
        subplot(n_rows, n_cols, j)
        eda(Clusterings_VCA_I_KDE.Labels(:,t))
        title({'M-LUND+VCA-I+KDE Clustering'; strcat('$K=', num2str(Clusterings_VCA_I_KDE.K(t)), '$, $\log_2(t)=', num2str(log2(Clusterings_VCA_I_KDE.TimeSamples(t))), '$, ')}, 'interpreter', 'latex')
        colorbar off
        set(gca,'FontSize', 16)
    
    end
   
    
    saveas(fig, strcat(file_name(1:end-4), '-M-LUND+VCA-I+KDE'), 'jpeg')        
    
    
end
close all