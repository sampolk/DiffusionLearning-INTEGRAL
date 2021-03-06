function [acc,labels,t] = atgp_lund(X, k, h, w, G, T, Y, flag)

% Input:
%      - X: p*N matrix, each column represent a pixel
%      - k: Number of endmembers
%      - h, w: height and width of the 3D datacube
%      - G: Graph constructed by extract_graph function
%      - T: A vector of power of diffusion time
%      - Y: GT label, N*1
%      - flag: default as 0. For Salinas A dataset, 1 means that the bare
%              soil pixels are masked

% Output:
%      - acc: Overall accuracy
%      - labels: The labels match up with the ground truth
%      - t: diffision time



acc = zeros(length(T),1);
t = acc;



if flag == 0
    
    labels = zeros(length(Y),length(T));
    
    for i = 1:length(T)
        [ATGP,~] = EIA_ATGP(X,k);
        [X_atgp] = hyperNnls(X, ATGP);
        X_atgp = reshape(X_atgp', h, w, k);
    
        

        p = compute_purity(X_atgp,83,86,k);
        
        [acc(i),labels(:,i),~,t(i),~] = LUND_labels(X, Y, p, G, T(i), h, w, k, 0);
    end
    
elseif flag == 1
    
    labels = zeros(length(Y(Y~=1)),length(T));
    for i = 1:length(T)
        [ATGP,~] = EIA_ATGP(X(:,Y~=1),k);
        [X_atgp] = hyperNnls(X, ATGP);
        X_atgp = reshape(X_atgp', h, w, k);
    
        p = compute_purity(X_atgp,83,86,k);
        [acc(i),labels(:,i),~,t(i),~] = LUND_labels(X, Y, p, G, T(i), h, w, k, 1);
    end
end


end