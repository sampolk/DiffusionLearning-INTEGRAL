%% Build Synthetic Dataset

% 10x25 slot in upper left corner of HSI
X1 = exp(-sqrt(-1)*2*pi.*randn(10*25,1));
X1 = [real(X1).*(1+0.05*randn(10*25,1)),imag(X1).*(1+0.05*randn(10*25,1))]+1.5;

% 20x20 slot in lower right corner of HSI
X2 = exp(-sqrt(-1)*2*pi*randn(20*20,1));
X2 = 0.05.*[real(X2).*(1+0.05*randn(length(X2),1)),imag(X2).*(1+0.05*randn(length(X2),1))]+1.5;


HSI = 0.05.*randn(50,50,2);
HSI(1:10,1:25,:) = reshape(X1, 10,25,2) + HSI(1:10, 1:25,:);
HSI(31:end, 31:end,:) = reshape(X2, 20,20,2) + HSI(31:50, 31:50,:);
GT = ones(50,50);
GT(1:10,1:25) = GT(1:10,1:25) +1;
GT(31:end, 31:end) = GT(31:end, 31:end) +2;

clear X1 X2;

X = reshape(HSI, 50*50, 2);
Y = reshape(GT,50*50,1);

subplot(1,2,1)
imagesc(GT)
axis equal tight 
title('Ground Truth Image Labels') 


subplot(1,2,2)
scatter(X(:,1), X(:,2), 36, Y, 'filled')
axis equal tight
title('Nonlinear Structure of Image ') 

[Idx_NN, Dist_NN] = knnsearch(X,X, 'K', length(X));
Idx_NN = Idx_NN(:,2:end); 
Dist_NN = Dist_NN(:,2:end);
n = length(X);
%%  Set Hyperparameters

Hyperparameters.SpatialParams.ImageSize = [50,50];
Hyperparameters.DiffusionNN = 1000;
Hyperparameters.DensityNN = 10;
Hyperparameters.WeightType = 'gaussian';
Hyperparameters.Sigma = 0.5;
Hyperparameters.AbundanceSigma = 0.4;
Hyperparameters.AbundanceDiffusionNN = 20;
Hyperparameters.NEigs = 4; % We truncate diffusion map to the first 3 eigenvectors later.
Hyperparameters.DiffusionTime = 0;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.Beta = 2;
Hyperparameters.Tau = 10^(-5);
Hyperparameters.Tolerance = 10^(-8);
Hyperparameters.K_Known = 3;

%% Extract Graph 

G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

% Plot eigenvectors from graph constructed from X
n_plots = Hyperparameters.K_Known-1;
[~,i] = sort(Y);
for k = 1:n_plots
 
    subplot(n_plots,n_plots, sub2ind([n_plots,n_plots], k,k))
    imagesc(log10(squareform(pdist(G.EigenVecs(i,k+1)))))
    
    title(strcat('$\log_{10}(||\psi_{i,', num2str(k+1), '}-\psi_{i,', num2str(k+1), '}||_2)$'), 'interpreter', 'latex')
    xlabel(strcat('Data index, $i$'), 'interpreter', 'latex')
    ylabel(strcat('Data index, $j$'), 'interpreter', 'latex')    
    pbaspect([1,1,1])
        
    for j = setdiff(1:n_plots,k)
        subplot(n_plots,n_plots, sub2ind([n_plots,n_plots], j,k))
        scatter(G.EigenVecs(:,k+1), G.EigenVecs(:,j+1), 36, Y, 'filled')
        
        xlabel(strcat('$(\psi_{', num2str(k+1), '})_i$'), 'interpreter', 'latex')
        ylabel(strcat('$(\psi_{', num2str(j+1), '})_i$'), 'interpreter', 'latex')
        pbaspect([1,1,1])
        
    end
end


%% Spectral Clustering as Initialization

close all

C = kmeans(G.EigenVecs(:,1:Hyperparameters.K_Known),Hyperparameters.K_Known);
t = floor(2.^(-log2(1-G.EigenVals(Hyperparameters.K_Known+1))));

scatter(X(:,1), X(:,2), 36, C, 'filled')
axis equal tight
title('Spectral Clustering Results')

disp('Performance of Spectral Clustering:')
disp(nmi(C,Y))

%% Extract Dictionary and Abundances From Diffusion Map


DiffusionMap = zeros(size(G.EigenVecs,1),2);

for k = 2:length(unique(Y))
    DiffusionMap(:,k-1) = G.EigenVecs(:,k).*(G.EigenVals(k)^t);
end

K = 3;
A = [];
for k = 1:K
    
    Xk = DiffusionMap(C==k, :);
    losses = zeros(size(DiffusionMap,2)-1,1);
    for c = 1:length(losses)
        try 
            [~, ~, ~, loss] = mvcnmf_shell(Xk, c+1);
            losses(c) = loss;
        catch
            % Didn't converge using Conjugate Gradient Descent. 
            % So, try standard gradient descent. 
            try 
                [~, ~, ~, loss] = mvcnmf_shell(Xk, c+1,2); 
                losses(c) = loss;
            catch
                losses(c) = NaN; % NMF Didn't converge for either algorithm. So, reject this choice of c
            end
        end
    end
    [~,c] = min(losses);
    [A_est, S_est, volume, loss] = mvcnmf_shell(Xk, c+1);
    A = [A; A_est];
end
M = MinSubset(A',1e-8);
A = hyperNnls(DiffusionMap', M)';
A = A./sum(A,2);
A(:, sum(A,1)==0) = []; % Get rid of empty columns

% Plot columns of A against each other
[~,i] = sort(Y);
for k = 1:size(A,2)
 
    subplot(size(A,2),size(A,2), sub2ind([size(A,2),size(A,2)], k,k))
    imagesc(log10(squareform(pdist(A(i,k)))))
    
    title(strcat('$\log_{10}(||A_{i,', num2str(k), '}-A_{j,', num2str(k), '}||_2)$'), 'interpreter', 'latex')
    xlabel(strcat('Data index, $i$'), 'interpreter', 'latex')
    ylabel(strcat('Data index, $j$'), 'interpreter', 'latex')    
    pbaspect([1,1,1])
        
    for j = setdiff(1:size(A,2),k)
        subplot(size(A,2),size(A,2), sub2ind([size(A,2),size(A,2)], j,k))
        scatter(A(:,k), A(:,j), 36, Y)
        
        xlabel(strcat('$A_{:,', num2str(k), '}$'), 'interpreter', 'latex')
        ylabel(strcat('$A_{:,', num2str(j), '}$'), 'interpreter', 'latex')

        
        pbaspect([1,1,1])
        
    end
end


%% Extract New Graph Using Abundances

[Idx, Dist] = knnsearch(A,A, 'K', n);
Idx = Idx(:,2:end); 
Dist = Dist(:,2:end);


Hyperparameters.DiffusionNN = Hyperparameters.AbundanceDiffusionNN;
Hyperparameters.Sigma = Hyperparameters.AbundanceSigma;


GAb = extract_graph_large(A, Hyperparameters, Idx, Dist);
sigma = Hyperparameters.Sigma;

% Plot eigenVectors from graph constructed from A

n_plots = 2;

[~,i] = sort(Y);
for k = 1:n_plots
 
    subplot(n_plots,n_plots, sub2ind([n_plots,n_plots], k,k))
    imagesc(log10(squareform(pdist(GAb.EigenVecs(i,k+1)))))
    
    title(strcat('$\log_{10}(||\psi_{i,', num2str(k+1), '}-\psi_{i,', num2str(k+1), '}||_2)$'), 'interpreter', 'latex')
    xlabel(strcat('Data index, $i$'), 'interpreter', 'latex')
    ylabel(strcat('Data index, $j$'), 'interpreter', 'latex')    
    pbaspect([1,1,1])
        
    for j = setdiff(1:n_plots,k)
        subplot(n_plots,n_plots, sub2ind([n_plots,n_plots], j,k))
        scatter(GAb.EigenVecs(:,k+1), GAb.EigenVecs(:,j+1), 36, Y, 'filled')
        
        xlabel(strcat('$(\psi_{', num2str(k+1), '})_i$'), 'interpreter', 'latex')
        ylabel(strcat('$(\psi_{', num2str(j+1), '})_i$'), 'interpreter', 'latex')
        pbaspect([1,1,1])
        
    end
end

%% Cluster using DVIS

Hyperparameters.DensityNN = 10;
Hyperparameters.Sigma0 = prctile(Dist_NN(:,1:Hyperparameters.DensityNN), 90,'all');

p = KDE_large(Dist_NN,Hyperparameters);

C = LearningbyUnsupervisedNonlinearDiffusion_large(A, Hyperparameters, 2^15, GAb, p);

disp('Performance of Updated D-VIS Clustering:')
disp(nmi(C,Y))
  
close all 
scatter(X(:,1), X(:,2), 36, C, 'filled')
axis equal tight
title('Proposed Algorithm Results')
