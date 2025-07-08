#-------------------Search Spaces for Bayesian Search-----------------

#ESN and Ring share the same search space
ESN_units_min_val = 50
ESN_units_max_val = 200
ESN_units_step = 10

ESN_leaky_min_val = 0.01
ESN_leaky_max_val = 0.99
ESN_leaky_step = 0.05
ESN_leaky_sampling = "linear"

ESN_input_scaling_min_val = 0.01
ESN_input_scaling_max_val = 10
ESN_input_scaling_step = None
ESN_input_scaling_sampling = "log"

ESN_bias_scaling_min_val = 0.001
ESN_bias_scaling_max_val = 1
ESN_bias_scaling_step = None
ESN_bias_scaling_sampling = "log"

ESN_spectral_radius_min_val = 0.01
ESN_spectral_radius_max_val = 0.99
ESN_spectral_radius_step = 0.05
ESN_spectral_radius_sampling = "linear"

#MF-ESN and MF-Ring share the same search space
MF_units_min_val = 50
MF_units_max_val = 200
MF_units_step = 10 

MF_input_scaling_min_val = 0.1
MF_input_scaling_max_val = 10
MF_input_scaling_step = None 
MF_input_scaling_sampling = "log"

MF_memory_factor_min_val = 0.01
MF_memory_factor_max_val = 0.99
MF_memory_factor_step = 0.05  
MF_memory_factor_sampling = "linear"

MF_bias_scaling_min_val = 0.001
MF_bias_scaling_max_val = 1
MF_bias_scaling_step = None 
MF_bias_scaling_sampling = "log"

MF_gamma_min_val = 0.1
MF_gamma_max_val = 1
MF_gamma_step = 0.05  
MF_gamma_sampling = "linear"


MF_p_min_val = 1
MF_p_max_val = 10
MF_p_step = 1  
MF_p_sampling = "linear"

MF_alpha_fixed_val = 1  

# Fixed physical parameters for Memristive Friendly architectures
MF_kp0_fixed_val = 0.0001
MF_kd0_fixed_val = 0.5
MF_etap_fixed_val = 10
MF_etad_fixed_val = 1

MF_dt_min_val = 0.0001
MF_dt_max_val = 0.1
MF_dt_step = None 
MF_dt_sampling = "log"

#readout layer regulariser value, tuned through the same bayestian search as the other parameters
readout_regularizer_min_val = 0.001
readout_regularizer_max_val = 1
readout_regularizer_step = None
readout_regularizer_sampling = "log"