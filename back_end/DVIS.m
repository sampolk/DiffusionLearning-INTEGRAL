function Clusterings = DVIS(X, G, density, Hyperparameters, C) 


purity = DVISpurity(G, Hyperparameters, C); % Calculate Geometric Purity
Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), purity./max(purity)],2));
