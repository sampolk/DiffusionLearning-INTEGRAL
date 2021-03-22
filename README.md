# DiffusionLearning-INTEGRAL

This toolbox allows the implementation of the following diffusion-based clustering algorithms on very large datasets:

 - Learning by Unsupervised Nonlinear Diffusion (LUND)
 - Multiscale Learning by Unsupervised Nonlinear Diffusion (M-LUND)

Once this toolbox is downloaded, please do the following procedure. It only needs to be done once.

  1. Ensure that the AVIRS-NG India Forest dataset (titled "r1_reg.mat") is in your MATLAB path. 
  2. Run "preprocessing.m". This script will save nearest neighbors and a standardized version of the image locally. This file is 8.07 GB. 

After the above is done, you can analyze the AVIRIS-NG India Forest dataset using the main.m file. 

For large datasets like the AVIRS-NG India Forest dataset, we are often limited by RAM. On my MacBook Pro with 8GB of RAM, I can only store weight matrices with 1000 nearest neighbors in my workspace. 

I also included code for a Python implementation of LUND that relies on the [ANNOY](https://github.com/spotify/annoy) package for fast approximate nearest neighbor searches. This code is included in a Jupyter Notebook called 'lund.ipynb'. To use this version of LUND, the following Python 3 packages must be installed: 

  1. NumPy
  2. SciPy
  3. Scikit-learn
  4. ANNOY: pip installation instructions given on [ANNOY GitHub page](https://github.com/spotify/annoy).
