# DiffusionLearning-INTEGRAL

This toolbox allows the implementation of the following diffusion-based clustering algorithms on very large datasets:

    - Learning by Unsupervised Nonlinear Diffusion (LUND)

The main.m script allows the user to generate LUND clusterings of the AVIRS-NG Indian Forest dataset. You must have run india_forest.m first and downloaded the file titled 'normalized_data_NNs.mat'. Otherwise, this script will not work. 

For large datasets like the AVIRS-NG India Forest dataset, we are often limited by RAM. On my MacBook Pro with 8GB of RAM, I can only store weight matrices with 1000 nearest neighbors in my workspace. 

All necessary data are contained in this repository, so no additional data downloads are necessary. 
