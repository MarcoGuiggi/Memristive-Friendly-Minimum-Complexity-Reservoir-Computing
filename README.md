# Memristive-Friendly-Minimum-Complexity-Reservoir-Computing
Code base for my thesis project. Contains code to set up and run benchmarking experiments on reservoir computing models of different complexities.

Memristive_Friendly_Minimum_Complexity_ESN.py contains all the models tested in the experiments:ESN, MF-ESN, RingESN, MF-RingESN, MinCompESN and MF-MinCompESN.

In the experiments_bayesian_search files is the code to run the experiments, which perform model selection through Bayesian search and then log the results of the best models found. The logs are thorough and create a new file for each experiment run. 

Bayesian_Search_Hypermodels.py defines the hyperparameter ranges for the model selection of every model.

The critical_difference_plot files have been used to produce the Critical Difference Diagrams (CD) that appear in the thesis paper. Here they are already loaded with the results listed in the thesis paper.
