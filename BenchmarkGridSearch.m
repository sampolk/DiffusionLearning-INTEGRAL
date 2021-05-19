%% Specify Grid Search

clc
clear
 
profile off;
profile on;

prompt = 'Which dataset? \n 1) Indian Pines \n 2) Indian Pines (Corrected) \n 3) Pavia Centre \n 4) Pavia University \n 5) Salinas \n 6) Salinas (Corrected) \n 7) Salinas A \n 8) Salinas A (Corrected) \n 9) Botswana \n 10) Kennedy Space Center \n';
DataSelected = input(prompt);

if DataSelected == 1
    load('IndianPines.mat')
elseif DataSelected ==2
    load('IndianPinesCorrected.mat')
elseif DataSelected == 3 
    load('PaviaCentre.mat')
    Dist_NN = [D1, D2, D3, D4];
    Idx_NN  = [I1, I2, I3, I4];
elseif DataSelected == 4
    load('PaviaU.mat')
elseif DataSelected == 5
    load('Salinas.mat')
elseif DataSelected == 6
    load('SalinasCorrected.mat')
elseif DataSelected == 7
    load('SalinasA.mat')
elseif DataSelected == 8
    load('SalinasACorrected.mat')
elseif DataSelected == 9
    load('Botswana.mat')
    Dist_NN = [D1, D2, D3, D4];
    Idx_NN  = [I1, I2, I3, I4];
elseif DataSelected == 10
    load('KennedySpaceCenter.mat')
    Dist_NN = [D1, D2, D3, D4];
    Idx_NN  = [I1, I2, I3, I4];    
else
    disp('Incorrect prompt input. Please enter one of [1:10].')
end
Dist_NN = Dist_NN(:,2:end);
Idx_NN = Idx_NN(:,2:end);

if norm(single(Dist_NN==0))>0 
    minval =  min(Dist_NN(Dist_NN>0));
    for j = 1:size(Dist_NN,2)
        Dist_NN(Dist_NN(:,j)==0,j) = minval/10;
    end
end

% Set Default parameters
Hyperparameters.SpatialParams.ImageSize = [M,N];
Hyperparameters.NEigs = 10;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.Beta = 2;
Hyperparameters.Tau = 10^(-5);

clc
prompt = 'Which Endmember Extraction Method? \n 1) ATGP \n 2) VCA \n 3) N-FINDR \n';
AlgSelected = input(prompt);

if AlgSelected == 1
    
    Hyperparameters.EndmemberParams.Algorithm = 'ATGP';
    Hyperparameters.K_Known = length(unique(Y));
    Hyperparameters.EndmemberParams.K = length(unique(Y));
    
elseif AlgSelected == 2

    if DataSelected<=2
        Hyperparameters.EndmemberParams.Algorithm = 'VCA';
        Hyperparameters.K_Known = 16;
        Hyperparameters.EndmemberParams.K = 16;
    elseif DataSelected>= 5 && DataSelected<=8
        Hyperparameters.EndmemberParams.Algorithm = 'VCA-I';
        Hyperparameters.K_Known = length(unique(Y));
        Hyperparameters.EndmemberParams.K = length(unique(Y));
    else
        Hyperparameters.EndmemberParams.Algorithm = 'VCA';
        Hyperparameters.K_Known = length(unique(Y));
        Hyperparameters.EndmemberParams.K = length(unique(Y));
    end
    
elseif AlgSelected == 3
    
    if DataSelected<=2
        Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR';
        Hyperparameters.K_Known = 16;
        Hyperparameters.EndmemberParams.K = 16;
    elseif DataSelected>= 5 && DataSelected<=8
        Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR-I';
        Hyperparameters.K_Known = length(unique(Y));
        Hyperparameters.EndmemberParams.K = length(unique(Y));
    else
        Hyperparameters.EndmemberParams.Algorithm = 'N-FINDR';
        Hyperparameters.K_Known = length(unique(Y));
        Hyperparameters.EndmemberParams.K = length(unique(Y));
    end
end

clc
prompt = 'Should the number of clusters and number of endmembers be learned from the data or ground truth? \n 1) Learned from data \n 2) Learned from ground truth \n';
SupSelected = input(prompt);

if SupSelected==1
    Hyperparameters = rmfield(Hyperparameters, 'K_Known');
    % Use HySime to calculate Optimal No. Endmembers.
    [kf,~]=hysime(reshape(HSI,M*N,size(HSI,3))'); 
    Hyperparameters.EndmemberParams.K = kf;
end

clc
if SupSelected == 2
    prompt = 'Should we compare against benchmark algorithms? \n 1) Yes \n 2) No \n';
    CompareSelected = input(prompt);

    if CompareSelected == 1
        compare_on = 1;
    elseif CompareSelected == 2
        compare_on = 0;
    else
        disp('Incorrect prompt input. Please enter one of [1,2].')
    end
else
    compare_on = 0;
end

clc
prompt = 'Should we save all our results? \n 1) Yes \n 2) No \n';
SaveSelected = input(prompt);

if SaveSelected == 1
    save_on = 1;
elseif SaveSelected == 2
    save_on = 0;
else
    disp('Incorrect prompt input. Please enter one of [1,2].')
end

clc
if ~isempty(intersect(DataSelected, [7, 8]))
    disp('Performing Nearest Neighbor Searches...')

    X = X + randn(size(X)).*10^(-7);

    [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);

    Dist_NN = Dist_NN(:,2:end);
    Idx_NN = Idx_NN(:,2:end);
end

disp('Extracting Endmembers...')
[PixelPurity, endmembers, abundances] = compute_purity(X,Hyperparameters);

[Idx, Dist] = knnsearch(abundances, abundances, 'K', 1000);

Idx_Ab = Idx(:,2:end);
Dist_Ab = Dist(:,2:end);

profile off

clear AlgSelected DataSelected Dist Idx kf prompt SupSelected CompareSelected SaveSelected
clc
disp('Dataset Preloaded.')
disp('Set hyperparameter grid below to begin grid search')

%% Set Hyperparameter Grid

% Set number of nearest neighbors to use in graph and KDE construction.
NNs = setdiff(unique([5,round(10.^(1:0.3:3),-1),999]), [1000]); 

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles =0:10:100;

%% Run Grid Search

nmis = zeros(length(NNs), length(prctiles), 3 + compare_on);
tic
for i = 1:length(NNs)
    
    Hyperparameters.DiffusionNN = NNs(i);
    Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
        
    if compare_on
        % Evaluate Spectral Clustering on graph
        
        n_eigs = Hyperparameters.NEigs;
        % Spectral Clustering
        Hyperparameters.NEigs = Hyperparameters.K_Known;
        G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
        if G.EigenVals(2)<1
            nmis(i,:,4) = nmi(SpectralClustering(G,Hyperparameters.K_Known),Y);
        else
            nmis(i,:,4) = NaN;
        end
        Hyperparameters.NEigs = n_eigs; % Reset no. eigenvalues
    end
    
    Hyperparameters.IncludeDensity = 0;
    G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

    % Compute Clusterings Using Solely Pixel Purity
    Hyperparameters.IncludeDensity = 0;
    
    Clusterings = MLUNDEndmember(X, Hyperparameters, Idx_NN, Dist_NN, G, PixelPurity, endmembers);
    performance = measure_performance(Clusterings, Y);
    nmis(i,:,2) = performance(1);
    Hyperparameters.IncludeDensity = 1;

    if save_on
        save(strcat('M-LUND_VCA-', num2str(NNs(i)),'.mat'), 'Clusterings', 'performance' )
    end
    
    
    if G.EigenVals(2)<1   
         
        for j = 1:length(prctiles)
            
            % Compute KDE using original dataset
            Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN), prctiles(j), 'all');
 
            if Hyperparameters.Sigma0>0
                 
                % M-LUND In Its Original Form - Just uses KDE
                Clusterings = MLUND_large(X, Hyperparameters, G, KDE_large(Dist_NN, Hyperparameters));
                performance1 = measure_performance(Clusterings, Y);
                nmis(i,j,1) = performance1(1);
 
                 % Compute Clusterings Using VCA Endmember Purity and KDE
                Hyperparameters.IncludeDensity = 1;
                Clusterings = MLUNDEndmember(X, Hyperparameters, Idx_NN, Dist_NN, G, PixelPurity, endmembers);
                performance = measure_performance(Clusterings, Y);        
                nmis(i,j,3)  = performance(1);

                if save_on
                    save(strcat('M-LUND_', num2str(NNs(i)), '-',num2str(prctiles(j)), '.mat'), 'Clusterings', 'performance' )
                    save(strcat('M-LUND_VCA_I_KDE_', num2str(NNs(i)), '-',num2str(prctiles(j)),'.mat'), 'Clusterings', 'performance1')
                end

                t = toc;
                if toc>20
                    disp(strcat('NNs:', num2str(floor((i-1)/length(NNs)*100)), '% Complete. prctiles:',num2str(floor(j/length(prctiles)*100)), '% Complete.'))
                    tic
                end
                
            else
                nmis(i,j,:) = NaN;
            end
        end
    else
        nmis(i,:,:) = NaN;  
    end
end
        
if compare_on
    
    nmi_summary = zeros(1,8);    
    
    % K-Means Clustering
    nmi_summary(1) = nmi( kmeans(X, Hyperparameters.K_Known),Y);

    % K-Means Clustering after PCA.
    [~,b,c] = pca(X);
    nmi_summary(2) = nmi(kmeans(b(:,1:find(cumsum(c)/sum(c)>0.99,1, 'first')), Hyperparameters.K_Known),Y);

    % Gaussian Mixture Model, fit to PCA Data
    try
        nmi_summary(3) = nmi(cluster(fitgmdist(b(:,1:find(cumsum(c)/sum(c)>0.99,1, 'first')), Hyperparameters.K_Known), b(:,1:find(cumsum(c)/sum(c)>0.99,1, 'first'))),Y);
    catch
        try
            nmi_summary(3) = nmi(cluster(fitgmdist(b(:,1:find(cumsum(c)/sum(c)>0.95,1, 'first')), Hyperparameters.K_Known), b(:,1:find(cumsum(c)/sum(c)>0.95,1, 'first'))),Y);
        catch
            try
                nmi_summary(3) = nmi(cluster(fitgmdist(b(:,1:find(cumsum(c)/sum(c)>0.9,1, 'first')), Hyperparameters.K_Known), b(:,1:find(cumsum(c)/sum(c)>0.9,1, 'first'))),Y);
            catch
                nmi_summary(3) = NaN;
            end
        end
    end
    
    % DBSCAN    
    nmi_summary(4) = nmi( dbscan(X,median(Dist_NN(:,1:20),'all'),20), Y);
    
    % Spectral Clustering
    nmi_summary(5) =  max(nmis(:,:,4),[],'all');
    
    for k = 1:3
        nmi_summary(5+k) = max(nmis(:,:,k),[],'all');
    end
    
    variable_names = {'KMeans', 'K-MeansPCA', 'GMMPCA', 'DBSCAN', 'SC', 'MLUND', 'MLUNDVCA', 'MLUNDVCAKDE'};
    nmi_summary = array2table(nmi_summary, 'VariableNames', variable_names);
else
    nmi_summary = zeros(1,3);
    for k = 1:3
        nmi_summary(k) = max(nmis(:,:,k),[],'all');
    end

    variable_names = { 'MLUND', 'MLUNDVCA', 'MLUNDVCAKDE'};
    nmi_summary = array2table(nmi_summary, 'VariableNames', variable_names);
end
    
save('summary_statistics', 'nmis', 'NNs', 'prctiles', 'nmi_summary')

disp(nmi_summary)