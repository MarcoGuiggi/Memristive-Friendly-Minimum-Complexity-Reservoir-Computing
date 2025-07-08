from Bayesian_Search_Space import *
import keras_tuner
from Memristive_Friendly_Minimum_Complexity_ESN import *

#------------------------- Classification Models ---------------------------


class HyperESN_classification(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(name = "num_units", min_value=ESN_units_min_val, max_value=ESN_units_max_val, step=ESN_units_step)
        leaky = hp.Float(name = "leaky", min_value = ESN_leaky_min_val, max_value = ESN_leaky_max_val, step=ESN_leaky_step, sampling=ESN_leaky_sampling)
        input_scaling = hp.Float(name="input_scaling", min_value=ESN_input_scaling_min_val, max_value=ESN_input_scaling_max_val, step=ESN_input_scaling_step, sampling=ESN_input_scaling_sampling)
        bias_scaling = hp.Float(name="bias_scaling", min_value=ESN_bias_scaling_min_val, max_value=ESN_bias_scaling_max_val, step=ESN_bias_scaling_step, sampling=ESN_bias_scaling_sampling)
        spectral_radius = hp.Float(name="spectral_radius", min_value=ESN_spectral_radius_min_val, max_value=ESN_spectral_radius_max_val, step=ESN_spectral_radius_step, sampling=ESN_spectral_radius_sampling)
        redout_regularizer = hp.Float(name = "readout_regularizer", min_value=readout_regularizer_min_val, max_value=readout_regularizer_max_val, step=readout_regularizer_step, sampling=readout_regularizer_sampling)

        model = ESN(
                        units=units, leaky=leaky, 
                        input_scaling=input_scaling, bias_scaling=bias_scaling, 
                        spectral_radius=spectral_radius, readout_regularizer=redout_regularizer
                    )
        
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    
        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        #FARSI PASSARE LE VALIDATION DAL TUNER TRAMITE KWARGS E USARE QUELLE PER VALIDARE

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
    
class HyperRing_classification(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(name = "num_units", min_value=ESN_units_min_val, max_value=ESN_units_max_val, step=ESN_units_step)
        leaky = hp.Float(name = "leaky", min_value = ESN_leaky_min_val, max_value = ESN_leaky_max_val, step=ESN_leaky_step, sampling=ESN_leaky_sampling)
        input_scaling = hp.Float(name="input_scaling", min_value=ESN_input_scaling_min_val, max_value=ESN_input_scaling_max_val, step=ESN_input_scaling_step, sampling=ESN_input_scaling_sampling)
        bias_scaling = hp.Float(name="bias_scaling", min_value=ESN_bias_scaling_min_val, max_value=ESN_bias_scaling_max_val, step=ESN_bias_scaling_step, sampling=ESN_bias_scaling_sampling)
        spectral_radius = hp.Float(name="spectral_radius", min_value=ESN_spectral_radius_min_val, max_value=ESN_spectral_radius_max_val, step=ESN_spectral_radius_step, sampling=ESN_spectral_radius_sampling)
        redout_regularizer = hp.Float(name = "readout_regularizer", min_value=readout_regularizer_min_val, max_value=readout_regularizer_max_val, step=readout_regularizer_step, sampling=readout_regularizer_sampling)

        model = RingESN(
                        units=units, leaky=leaky, 
                        input_scaling=input_scaling, bias_scaling=bias_scaling, 
                        spectral_radius=spectral_radius, readout_regularizer=redout_regularizer
                    )
        
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    

        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) 

class HyperMF_ESN_classification(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(
            name="num_units",
            min_value=MF_units_min_val,
            max_value=MF_units_max_val,
            step=MF_units_step
        )
        input_scaling = hp.Float(
            name="input_scaling",
            min_value=MF_input_scaling_min_val,
            max_value=MF_input_scaling_max_val,
            step=MF_input_scaling_step,
            sampling=MF_input_scaling_sampling
        )
        memory_factor = hp.Float(
            name="memory_factor",
            min_value=MF_memory_factor_min_val,
            max_value=MF_memory_factor_max_val,
            step=MF_memory_factor_step,
            sampling=MF_memory_factor_sampling
        )
        bias_scaling = hp.Float(
            name="bias_scaling",
            min_value=MF_bias_scaling_min_val,
            max_value=MF_bias_scaling_max_val,
            step=MF_bias_scaling_step,
            sampling=MF_bias_scaling_sampling
        )
        gamma = hp.Float(
            name="gamma",
            min_value=MF_gamma_min_val,
            max_value=MF_gamma_max_val,
            step=MF_gamma_step,
            sampling=MF_gamma_sampling
        )
        p = hp.Int(
            name="p",
            min_value=MF_p_min_val,
            max_value=MF_p_max_val,
            step=MF_p_step,
            sampling=MF_p_sampling
        )
        dt = hp.Float(
            name="dt",
            min_value=MF_dt_min_val,
            max_value=MF_dt_max_val,
            step=MF_dt_step,
            sampling=MF_dt_sampling
        )
        readout_regularizer = hp.Float(
            name = "readout_regularizer", 
            min_value=readout_regularizer_min_val, 
            max_value=readout_regularizer_max_val, 
            step=readout_regularizer_step, 
            sampling=readout_regularizer_sampling
        )

        #fixed values
        alpha = hp.Fixed(
            name="alpha",
            value=MF_alpha_fixed_val
        )
        kp0 = hp.Fixed(
            name="kp0",
            value=MF_kp0_fixed_val
        )
        kd0 = hp.Fixed(
            name="kd0",
            value=MF_kd0_fixed_val
        )
        etap = hp.Fixed(
            name="etap",
            value=MF_etap_fixed_val
        )
        etad = hp.Fixed(
            name="etad",
            value=MF_etad_fixed_val
        )

        # Instantiate the model with both tunable and fixed parameters
        model = MF(
            units=units,
            input_scaling=input_scaling,
            memory_factor=memory_factor,
            bias_scaling=bias_scaling,
            gamma=gamma,
            p=p,
            dt=dt,
            alpha=alpha,
            kp0=kp0,
            kd0=kd0,
            etap=etap,
            etad=etad,
            readout_regularizer=readout_regularizer
        )
    
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    
        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        #FARSI PASSARE LE VALIDATION DAL TUNER TRAMITE KWARGS E USARE QUELLE PER VALIDARE

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val)
    
class HyperMF_Ring_classification(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(
            name="num_units",
            min_value=MF_units_min_val,
            max_value=MF_units_max_val,
            step=MF_units_step
        )
        input_scaling = hp.Float(
            name="input_scaling",
            min_value=MF_input_scaling_min_val,
            max_value=MF_input_scaling_max_val,
            step=MF_input_scaling_step,
            sampling=MF_input_scaling_sampling
        )
        memory_factor = hp.Float(
            name="memory_factor",
            min_value=MF_memory_factor_min_val,
            max_value=MF_memory_factor_max_val,
            step=MF_memory_factor_step,
            sampling=MF_memory_factor_sampling
        )
        bias_scaling = hp.Float(
            name="bias_scaling",
            min_value=MF_bias_scaling_min_val,
            max_value=MF_bias_scaling_max_val,
            step=MF_bias_scaling_step,
            sampling=MF_bias_scaling_sampling
        )
        gamma = hp.Float(
            name="gamma",
            min_value=MF_gamma_min_val,
            max_value=MF_gamma_max_val,
            step=MF_gamma_step,
            sampling=MF_gamma_sampling
        )
        p = hp.Int(
            name="p",
            min_value=MF_p_min_val,
            max_value=MF_p_max_val,
            step=MF_p_step,
            sampling=MF_p_sampling
        )
        dt = hp.Float(
            name="dt",
            min_value=MF_dt_min_val,
            max_value=MF_dt_max_val,
            step=MF_dt_step,
            sampling=MF_dt_sampling
        )

        #fixed values
        alpha = hp.Fixed(
            name="alpha",
            value=MF_alpha_fixed_val
        )
        kp0 = hp.Fixed(
            name="kp0",
            value=MF_kp0_fixed_val
        )
        kd0 = hp.Fixed(
            name="kd0",
            value=MF_kd0_fixed_val
        )
        etap = hp.Fixed(
            name="etap",
            value=MF_etap_fixed_val
        )
        etad = hp.Fixed(
            name="etad",
            value=MF_etad_fixed_val
        )

        readout_regularizer = hp.Float(
            name = "readout_regularizer", 
            min_value=readout_regularizer_min_val, 
            max_value=readout_regularizer_max_val, 
            step=readout_regularizer_step, 
            sampling=readout_regularizer_sampling
        )

        # Instantiate the model with both tunable and fixed parameters
        model = MFRingESN(
            units=units,
            input_scaling=input_scaling,
            memory_factor=memory_factor,
            bias_scaling=bias_scaling,
            gamma=gamma,
            p=p,
            dt=dt,
            alpha=alpha,
            kp0=kp0,
            kd0=kd0,
            etap=etap,
            etad=etad,
            readout_regularizer=readout_regularizer
        )

        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)  

        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        if validation_data is not None:
            x_val, y_val = validation_data

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
  

#------------------------- Regression Models --------------------------------
class HyperESN_regression(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(name = "num_units", min_value=ESN_units_min_val, max_value=ESN_units_max_val, step=ESN_units_step)
        leaky = hp.Float(name = "leaky", min_value = ESN_leaky_min_val, max_value = ESN_leaky_max_val, step=ESN_leaky_step, sampling=ESN_leaky_sampling)
        input_scaling = hp.Float(name="input_scaling", min_value=ESN_input_scaling_min_val, max_value=ESN_input_scaling_max_val, step=ESN_input_scaling_step, sampling=ESN_input_scaling_sampling)
        bias_scaling = hp.Float(name="bias_scaling", min_value=ESN_bias_scaling_min_val, max_value=ESN_bias_scaling_max_val, step=ESN_bias_scaling_step, sampling=ESN_bias_scaling_sampling)
        spectral_radius = hp.Float(name="spectral_radius", min_value=ESN_spectral_radius_min_val, max_value=ESN_spectral_radius_max_val, step=ESN_spectral_radius_step, sampling=ESN_spectral_radius_sampling)
        redout_regularizer = hp.Float(name = "readout_regularizer", min_value=readout_regularizer_min_val, max_value=readout_regularizer_max_val, step=readout_regularizer_step, sampling=readout_regularizer_sampling)

        model = ESNregr(
                        units=units, leaky=leaky, 
                        input_scaling=input_scaling, bias_scaling=bias_scaling, 
                        spectral_radius=spectral_radius, readout_regularizer=redout_regularizer
                    )
        
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    
        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        #FARSI PASSARE LE VALIDATION DAL TUNER TRAMITE KWARGS E USARE QUELLE PER VALIDARE

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
    
class HyperRing_regression(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(name = "num_units", min_value=ESN_units_min_val, max_value=ESN_units_max_val, step=ESN_units_step)
        leaky = hp.Float(name = "leaky", min_value = ESN_leaky_min_val, max_value = ESN_leaky_max_val, step=ESN_leaky_step, sampling=ESN_leaky_sampling)
        input_scaling = hp.Float(name="input_scaling", min_value=ESN_input_scaling_min_val, max_value=ESN_input_scaling_max_val, step=ESN_input_scaling_step, sampling=ESN_input_scaling_sampling)
        bias_scaling = hp.Float(name="bias_scaling", min_value=ESN_bias_scaling_min_val, max_value=ESN_bias_scaling_max_val, step=ESN_bias_scaling_step, sampling=ESN_bias_scaling_sampling)
        spectral_radius = hp.Float(name="spectral_radius", min_value=ESN_spectral_radius_min_val, max_value=ESN_spectral_radius_max_val, step=ESN_spectral_radius_step, sampling=ESN_spectral_radius_sampling)
        redout_regularizer = hp.Float(name = "readout_regularizer", min_value=readout_regularizer_min_val, max_value=readout_regularizer_max_val, step=readout_regularizer_step, sampling=readout_regularizer_sampling)

        model = RingESNregr(
                        units=units, leaky=leaky, 
                        input_scaling=input_scaling, bias_scaling=bias_scaling, 
                        spectral_radius=spectral_radius, readout_regularizer=redout_regularizer
                    )
        
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    
        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        #FARSI PASSARE LE VALIDATION DAL TUNER TRAMITE KWARGS E USARE QUELLE PER VALIDARE

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!

class HyperMF_ESN_regression(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(
            name="num_units",
            min_value=MF_units_min_val,
            max_value=MF_units_max_val,
            step=MF_units_step
        )
        input_scaling = hp.Float(
            name="input_scaling",
            min_value=MF_input_scaling_min_val,
            max_value=MF_input_scaling_max_val,
            step=MF_input_scaling_step,
            sampling=MF_input_scaling_sampling
        )
        memory_factor = hp.Float(
            name="memory_factor",
            min_value=MF_memory_factor_min_val,
            max_value=MF_memory_factor_max_val,
            step=MF_memory_factor_step,
            sampling=MF_memory_factor_sampling
        )
        bias_scaling = hp.Float(
            name="bias_scaling",
            min_value=MF_bias_scaling_min_val,
            max_value=MF_bias_scaling_max_val,
            step=MF_bias_scaling_step,
            sampling=MF_bias_scaling_sampling
        )
        gamma = hp.Float(
            name="gamma",
            min_value=MF_gamma_min_val,
            max_value=MF_gamma_max_val,
            step=MF_gamma_step,
            sampling=MF_gamma_sampling
        )
        p = hp.Int(
            name="p",
            min_value=MF_p_min_val,
            max_value=MF_p_max_val,
            step=MF_p_step,
            sampling=MF_p_sampling
        )
        dt = hp.Float(
            name="dt",
            min_value=MF_dt_min_val,
            max_value=MF_dt_max_val,
            step=MF_dt_step,
            sampling=MF_dt_sampling
        )
        readout_regularizer = hp.Float(
            name = "readout_regularizer", 
            min_value=readout_regularizer_min_val, 
            max_value=readout_regularizer_max_val, 
            step=readout_regularizer_step, 
            sampling=readout_regularizer_sampling
        )

        #fixed values
        alpha = hp.Fixed(
            name="alpha",
            value=MF_alpha_fixed_val
        )
        kp0 = hp.Fixed(
            name="kp0",
            value=MF_kp0_fixed_val
        )
        kd0 = hp.Fixed(
            name="kd0",
            value=MF_kd0_fixed_val
        )
        etap = hp.Fixed(
            name="etap",
            value=MF_etap_fixed_val
        )
        etad = hp.Fixed(
            name="etad",
            value=MF_etad_fixed_val
        )

        # Instantiate the model with both tunable and fixed parameters
        model = MFregr(
            units=units,
            input_scaling=input_scaling,
            memory_factor=memory_factor,
            bias_scaling=bias_scaling,
            gamma=gamma,
            p=p,
            dt=dt,
            alpha=alpha,
            kp0=kp0,
            kd0=kd0,
            etap=etap,
            etad=etad,
            readout_regularizer=readout_regularizer
        )
    
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    
        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        #FARSI PASSARE LE VALIDATION DAL TUNER TRAMITE KWARGS E USARE QUELLE PER VALIDARE

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
    
class HyperMF_Ring_regression(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(
            name="num_units",
            min_value=MF_units_min_val,
            max_value=MF_units_max_val,
            step=MF_units_step
        )
        input_scaling = hp.Float(
            name="input_scaling",
            min_value=MF_input_scaling_min_val,
            max_value=MF_input_scaling_max_val,
            step=MF_input_scaling_step,
            sampling=MF_input_scaling_sampling
        )
        memory_factor = hp.Float(
            name="memory_factor",
            min_value=MF_memory_factor_min_val,
            max_value=MF_memory_factor_max_val,
            step=MF_memory_factor_step,
            sampling=MF_memory_factor_sampling
        )
        bias_scaling = hp.Float(
            name="bias_scaling",
            min_value=MF_bias_scaling_min_val,
            max_value=MF_bias_scaling_max_val,
            step=MF_bias_scaling_step,
            sampling=MF_bias_scaling_sampling
        )
        gamma = hp.Float(
            name="gamma",
            min_value=MF_gamma_min_val,
            max_value=MF_gamma_max_val,
            step=MF_gamma_step,
            sampling=MF_gamma_sampling
        )
        p = hp.Int(
            name="p",
            min_value=MF_p_min_val,
            max_value=MF_p_max_val,
            step=MF_p_step,
            sampling=MF_p_sampling
        )
        dt = hp.Float(
            name="dt",
            min_value=MF_dt_min_val,
            max_value=MF_dt_max_val,
            step=MF_dt_step,
            sampling=MF_dt_sampling
        )

        #fixed values
        alpha = hp.Fixed(
            name="alpha",
            value=MF_alpha_fixed_val
        )
        kp0 = hp.Fixed(
            name="kp0",
            value=MF_kp0_fixed_val
        )
        kd0 = hp.Fixed(
            name="kd0",
            value=MF_kd0_fixed_val
        )
        etap = hp.Fixed(
            name="etap",
            value=MF_etap_fixed_val
        )
        etad = hp.Fixed(
            name="etad",
            value=MF_etad_fixed_val
        )

        readout_regularizer = hp.Float(
            name = "readout_regularizer", 
            min_value=readout_regularizer_min_val, 
            max_value=readout_regularizer_max_val, 
            step=readout_regularizer_step, 
            sampling=readout_regularizer_sampling
        )

        # Instantiate the model with both tunable and fixed parameters
        model = MFRingESNregr(
            units=units,
            input_scaling=input_scaling,
            memory_factor=memory_factor,
            bias_scaling=bias_scaling,
            gamma=gamma,
            p=p,
            dt=dt,
            alpha=alpha,
            kp0=kp0,
            kd0=kd0,
            etap=etap,
            etad=etad,
            readout_regularizer=readout_regularizer
        )

        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)  

        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        if validation_data is not None:
            x_val, y_val = validation_data

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
    
#----------------------- Classification Models with Minimum Complexity ----------------------
    
class HyperRing_MinComp_classification(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(name = "num_units", min_value=ESN_units_min_val, max_value=ESN_units_max_val, step=ESN_units_step)
        leaky = hp.Float(name = "leaky", min_value = ESN_leaky_min_val, max_value = ESN_leaky_max_val, step=ESN_leaky_step, sampling=ESN_leaky_sampling)
        input_scaling = hp.Float(name="input_scaling", min_value=ESN_input_scaling_min_val, max_value=ESN_input_scaling_max_val, step=ESN_input_scaling_step, sampling=ESN_input_scaling_sampling)
        bias_scaling = hp.Float(name="bias_scaling", min_value=ESN_bias_scaling_min_val, max_value=ESN_bias_scaling_max_val, step=ESN_bias_scaling_step, sampling=ESN_bias_scaling_sampling)
        spectral_radius = hp.Float(name="spectral_radius", min_value=ESN_spectral_radius_min_val, max_value=ESN_spectral_radius_max_val, step=ESN_spectral_radius_step, sampling=ESN_spectral_radius_sampling)
        redout_regularizer = hp.Float(name = "readout_regularizer", min_value=readout_regularizer_min_val, max_value=readout_regularizer_max_val, step=readout_regularizer_step, sampling=readout_regularizer_sampling)

        model = RingESN_MinComp(
                        units=units, leaky=leaky, 
                        input_scaling=input_scaling, bias_scaling=bias_scaling, 
                        spectral_radius=spectral_radius, readout_regularizer=redout_regularizer
                    )
        
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    

        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) 

class HyperMF_Ring_MinComp_classification(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(
            name="num_units",
            min_value=MF_units_min_val,
            max_value=MF_units_max_val,
            step=MF_units_step
        )
        input_scaling = hp.Float(
            name="input_scaling",
            min_value=MF_input_scaling_min_val,
            max_value=MF_input_scaling_max_val,
            step=MF_input_scaling_step,
            sampling=MF_input_scaling_sampling
        )
        memory_factor = hp.Float(
            name="memory_factor",
            min_value=MF_memory_factor_min_val,
            max_value=MF_memory_factor_max_val,
            step=MF_memory_factor_step,
            sampling=MF_memory_factor_sampling
        )
        bias_scaling = hp.Float(
            name="bias_scaling",
            min_value=MF_bias_scaling_min_val,
            max_value=MF_bias_scaling_max_val,
            step=MF_bias_scaling_step,
            sampling=MF_bias_scaling_sampling
        )
        gamma = hp.Float(
            name="gamma",
            min_value=MF_gamma_min_val,
            max_value=MF_gamma_max_val,
            step=MF_gamma_step,
            sampling=MF_gamma_sampling
        )
        p = hp.Int(
            name="p",
            min_value=MF_p_min_val,
            max_value=MF_p_max_val,
            step=MF_p_step,
            sampling=MF_p_sampling
        )
        dt = hp.Float(
            name="dt",
            min_value=MF_dt_min_val,
            max_value=MF_dt_max_val,
            step=MF_dt_step,
            sampling=MF_dt_sampling
        )

        #fixed values
        alpha = hp.Fixed(
            name="alpha",
            value=MF_alpha_fixed_val
        )
        kp0 = hp.Fixed(
            name="kp0",
            value=MF_kp0_fixed_val
        )
        kd0 = hp.Fixed(
            name="kd0",
            value=MF_kd0_fixed_val
        )
        etap = hp.Fixed(
            name="etap",
            value=MF_etap_fixed_val
        )
        etad = hp.Fixed(
            name="etad",
            value=MF_etad_fixed_val
        )

        readout_regularizer = hp.Float(
            name = "readout_regularizer", 
            min_value=readout_regularizer_min_val, 
            max_value=readout_regularizer_max_val, 
            step=readout_regularizer_step, 
            sampling=readout_regularizer_sampling
        )

        # Instantiate the model with both tunable and fixed parameters
        model = MFRingESN_MinComp(
            units=units,
            input_scaling=input_scaling,
            memory_factor=memory_factor,
            bias_scaling=bias_scaling,
            gamma=gamma,
            p=p,
            dt=dt,
            alpha=alpha,
            kp0=kp0,
            kd0=kd0,
            etap=etap,
            etad=etad,
            readout_regularizer=readout_regularizer
        )

        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)  

        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        if validation_data is not None:
            x_val, y_val = validation_data

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
  

#------------------------- Regression Models with Minimum Complexity--------------------------------

class HyperRing_MinComp_regression(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(name = "num_units", min_value=ESN_units_min_val, max_value=ESN_units_max_val, step=ESN_units_step)
        leaky = hp.Float(name = "leaky", min_value = ESN_leaky_min_val, max_value = ESN_leaky_max_val, step=ESN_leaky_step, sampling=ESN_leaky_sampling)
        input_scaling = hp.Float(name="input_scaling", min_value=ESN_input_scaling_min_val, max_value=ESN_input_scaling_max_val, step=ESN_input_scaling_step, sampling=ESN_input_scaling_sampling)
        bias_scaling = hp.Float(name="bias_scaling", min_value=ESN_bias_scaling_min_val, max_value=ESN_bias_scaling_max_val, step=ESN_bias_scaling_step, sampling=ESN_bias_scaling_sampling)
        spectral_radius = hp.Float(name="spectral_radius", min_value=ESN_spectral_radius_min_val, max_value=ESN_spectral_radius_max_val, step=ESN_spectral_radius_step, sampling=ESN_spectral_radius_sampling)
        redout_regularizer = hp.Float(name = "readout_regularizer", min_value=readout_regularizer_min_val, max_value=readout_regularizer_max_val, step=readout_regularizer_step, sampling=readout_regularizer_sampling)

        model = RingESNregr_MinComp(
                        units=units, leaky=leaky, 
                        input_scaling=input_scaling, bias_scaling=bias_scaling, 
                        spectral_radius=spectral_radius, readout_regularizer=redout_regularizer
                    )
        
        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)    
        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        #FARSI PASSARE LE VALIDATION DAL TUNER TRAMITE KWARGS E USARE QUELLE PER VALIDARE

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            print("\n\nVALIDATION DATA NOT PASSED TO HYPERMODEL!!!\n\n")

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!

class HyperMF_Ring_MinComp_regression(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def build(self, hp):
        #takes the values defined in the search space section above, for ease of use
        units = hp.Int(
            name="num_units",
            min_value=MF_units_min_val,
            max_value=MF_units_max_val,
            step=MF_units_step
        )
        input_scaling = hp.Float(
            name="input_scaling",
            min_value=MF_input_scaling_min_val,
            max_value=MF_input_scaling_max_val,
            step=MF_input_scaling_step,
            sampling=MF_input_scaling_sampling
        )
        memory_factor = hp.Float(
            name="memory_factor",
            min_value=MF_memory_factor_min_val,
            max_value=MF_memory_factor_max_val,
            step=MF_memory_factor_step,
            sampling=MF_memory_factor_sampling
        )
        bias_scaling = hp.Float(
            name="bias_scaling",
            min_value=MF_bias_scaling_min_val,
            max_value=MF_bias_scaling_max_val,
            step=MF_bias_scaling_step,
            sampling=MF_bias_scaling_sampling
        )
        gamma = hp.Float(
            name="gamma",
            min_value=MF_gamma_min_val,
            max_value=MF_gamma_max_val,
            step=MF_gamma_step,
            sampling=MF_gamma_sampling
        )
        p = hp.Int(
            name="p",
            min_value=MF_p_min_val,
            max_value=MF_p_max_val,
            step=MF_p_step,
            sampling=MF_p_sampling
        )
        dt = hp.Float(
            name="dt",
            min_value=MF_dt_min_val,
            max_value=MF_dt_max_val,
            step=MF_dt_step,
            sampling=MF_dt_sampling
        )

        #fixed values
        alpha = hp.Fixed(
            name="alpha",
            value=MF_alpha_fixed_val
        )
        kp0 = hp.Fixed(
            name="kp0",
            value=MF_kp0_fixed_val
        )
        kd0 = hp.Fixed(
            name="kd0",
            value=MF_kd0_fixed_val
        )
        etap = hp.Fixed(
            name="etap",
            value=MF_etap_fixed_val
        )
        etad = hp.Fixed(
            name="etad",
            value=MF_etad_fixed_val
        )

        readout_regularizer = hp.Float(
            name = "readout_regularizer", 
            min_value=readout_regularizer_min_val, 
            max_value=readout_regularizer_max_val, 
            step=readout_regularizer_step, 
            sampling=readout_regularizer_sampling
        )

        # Instantiate the model with both tunable and fixed parameters
        model = MFRingESNregr_MinComp(
            units=units,
            input_scaling=input_scaling,
            memory_factor=memory_factor,
            bias_scaling=bias_scaling,
            gamma=gamma,
            p=p,
            dt=dt,
            alpha=alpha,
            kp0=kp0,
            kd0=kd0,
            etap=etap,
            etad=etad,
            readout_regularizer=readout_regularizer
        )

        return model
    
    def fit(self, hp, model, x, y, validation_data = None, **kwargs):

        #model is our custom ESN that we are tuning, we call its custom fit() method
        #and evaluate its result with our custom evaluate()
        model.fit(x, y, **kwargs)  

        #return custom evaluation on this fit(), the tuner handles averaging over multiple trials on the same hyperparameter combination
        if validation_data is not None:
            x_val, y_val = validation_data

        return model.evaluate(x_val, y_val) #!!!!!!!!!!!!!!!!!!!
    