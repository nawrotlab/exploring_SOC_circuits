# exploring_SOC_circuits
https://www.biorxiv.org/content/10.1101/2023.09.11.557174v2


Each run_xxx.py file allows to execute the respective experiment for the different models 1-5. Training odors and optional reward input can be specified as inputs to the respective run() function. 

Each run_modelx_overlap.py file allows to execute the respective experiment with overlapping training odors for the different models 1-5. 
Training odors, degree of odor overlap and optional reward input can be specified as inputs to the respective run() function.

Each run_modelx_overlap_test.py file allows to execute the respective experiment with overlapping test odors for the different models 1-5. 
Training odors, degree of odor overlap and optional reward input can be specified as inputs to the respective run() function.

All odor inputs are created in odors.py

models.py, models_var_connectivity.py and models_var_weights.py contain the model architectures. var_connectivity and var_weights refers to the KC-MBON connectivity matrix. 
