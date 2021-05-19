function C = SpectralClustering(G,K)

EigenVecs_Normalized = G.EigenVecs(:,1:K)./sqrt(sum(G.EigenVecs(1:K).^2,2));

C = kmeans(EigenVecs_Normalized, K);