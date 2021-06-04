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
    for j1 = 1:size(Dist_NN,2)
        Dist_NN(Dist_NN(:,j1)==0,j1) = minval/10;
    end
end

% Set Default parameters
Hyperparameters.SpatialParams.ImageSize = [M,N];
Hyperparameters.NEigs = 10;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.Beta = 2;
Hyperparameters.Tau = 10^(-5);
Hyperparameters.K_Known = length(unique(Y));
Hyperparameters.Tolerance = 1e-8;

clc
prompt = 'Should the number of clusters and number of endmembers be learned from the data or ground truth? \n 1) Learned from data \n 2) Learned from ground truth \n';
SupSelected = input(prompt);

if SupSelected==1
    Hyperparameters = rmfield(Hyperparameters, 'K_Known');
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
Hyperparameters.NEigs = max(length(unique(Y))+1, 10);


profile off

clear AlgSelected DataSelected Dist Idx kf prompt SupSelected CompareSelected SaveSelected
clc
disp('Dataset Preloaded.')
disp('Set hyperparameter grid below to begin grid search')

%% Set Hyperparameter Grid

% Set number of nearest neighbors to use in graph and KDE construction.
NNs_X = setdiff(unique([round(10.^(1.0:0.3:3),-1),999]), [1000]);

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 10:10:100;


%% Run Grid Search - Adjesency Matrix

compare_on = 1;
nmis = zeros(length(NNs_X), length(prctiles), 2 + compare_on);

for i1 = 1:length(NNs_X)
    
    Hyperparameters.DiffusionNN = NNs_X(i1);
    Hyperparameters.DensityNN = NNs_X(i1); % must be ≤ 1000
        
    % Extract Graph
    if isfield(Hyperparameters, 'K_Known')
        n_eigs = Hyperparameters.K_Known;
        if n_eigs>Hyperparameters.NEigs
            Hyperparameters.NEigs = Hyperparameters.K_Known;
        end
    else
        n_eigs = Hyperparameters.NEigs;
    end     
    
    GX = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    
    if GX.EigenVals(2)<1
        
        C_SC = SpectralClustering(GX,Hyperparameters.K_Known);
        if compare_on
            nmis(i1,:,3) = nmi(C_SC,Y);
        end
        GX.EigenVecs(:,n_eigs+2:end) = [];
        
        
        for j1 = 1:length(prctiles)
            
            % Compute KDE using original dataset
            Dist_temp = Dist_NN(:,1:Hyperparameters.DensityNN);            
            Hyperparameters.Sigma0 = prctile(Dist_temp(Dist_temp>0), prctiles(j1), 'all');
            density = KDE_large(Dist_NN, Hyperparameters);
            
            if sum(density == max(density)) < length(unique(Y))
                                
                % M-LUND In Its Original Form
                Clusterings = MLUND_large(X, Hyperparameters, GX, density);
                
                if ~isnan(Clusterings.K)
                    
                    [performance1,t] = measure_performance(Clusterings, Y);
                    C = Clusterings.Labels(:,t); % Best M-LUND Clustering
                    purity = DVISpurity(GX, Hyperparameters.Tolerance, C);
                    
                    Clusterings = MLUND_large(X, Hyperparameters, GX, harmmean([density./max(density), purity./max(purity)],2));
                    [performance2,~] = measure_performance(Clusterings, Y);
                    
                    nmis(i1,j1,1:2) = [performance1(1); performance2(1)];   
                    
                    disp(num2str([100*[i1-1,j1]./[length(NNs_X), length(prctiles)], performance1(1), performance2(1)],3))
                end
            end
        end
    end
end

compare_on = 1;
%%
if compare_on
    
    nmi_summary = zeros(1,7);    
    
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
    nmi_summary(5) =  max(nmis(:,:,3),[],'all');
    
    for k = 1:2
        nmi_summary(5+k) = max(nmis(:,:,k),[],'all');
    end
    
    variable_names = {'KMeans', 'K-MeansPCA', 'GMMPCA', 'DBSCAN', 'SC', 'MLUND', 'D-VIS'};
    nmi_summary = array2table(nmi_summary, 'VariableNames', variable_names);
else
    nmi_summary = zeros(1,2);
    for k = 1:2
        nmi_summary(k) = max(nmis(:,:,:,k),[],'all');
    end

    variable_names = { 'MLUND', 'D-VIS'};
    nmi_summary = array2table(nmi_summary, 'VariableNames', variable_names);
end
    
save('summary_statistics', 'nmis', 'NNs_X', 'prctiles','NNs_Ab', 'nmi_summary')

disp(nmi_summary)
