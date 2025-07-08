import numpy as np 
import tensorflow as tf
import tensorflow.keras as keras # type: ignore
from math import e
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import math_pi

    # default hyperparameters from Gianluca Milano
    # kp0 =  0.0001
    # kd0 =  0.5
    # eta_p =  10
    # eta_d =  1

# ------------------------------------ Z SCALING FUNCTIONS -------------------------------------

def scale_z(z, p):
    return (1.15-0.35) * (1 / (1 + np.exp(-z*p))) + 0.35

def scale_z2(z, p):
    return (1.15 - 0.35) * (1 / (1 + tf.exp(-z * p))) + 0.35

#------------------------------------ Aperiodic Weight Matrix------------------------------------------

#these functions are used to return a matrix for the input to reservoir connection weight
#the input layers is fully connected to the reservoir
#each input unit is assigned an aperiodic sequence created from teh digital expansion of pi
#as shown in the paper "Minimum Complexity Echo State Network" by Rodam et al

def get_signed_vector(length, start_digit, scaling):

    #returns 1D tensor filled with an aperiodic pattern of +1 and -1 based on the value of the first n digits of pi
    #math_pi.pi starts giving decimals from value a = 2, so we make sure we are always at least at position 2
    sequence_start_digit = start_digit + 2
    sequence_end_digit = start_digit + length

    #string with N digits of pi
    pi_n_digits_string = math_pi.pi(a = sequence_start_digit , b = sequence_end_digit+1)
    pi_digits = tf.constant(pi_n_digits_string)

    threshold = tf.constant(5, dtype=tf.int32) #digit between 0 and 4 -> -1, digit between 5 and 9 -> +1

    chars = tf.strings.bytes_split(pi_digits) #splits the string into an array with each character as byte element (al chars are digits here)
    digits = tf.strings.to_number(chars, out_type=tf.int32) #turns it into an array of int

    pi_digits_vector = tf.reshape(digits, [-1]) #creates a 1D tensor with the digits of pi

    mask = (pi_digits_vector >= threshold) #a boolean mask where for every element of the vector sets true or false depending on the condition

    true_vals  = tf.fill(dims=(length,), value=1)
    false_vals = tf.fill(dims=(length,), value=-1)

    signs_vector = tf.where(mask, true_vals, false_vals)

    signs_vector = tf.cast(signs_vector, tf.float32)

    weights_vector = tf.fill(dims=(length,), value=tf.cast(scaling, tf.float32))

    signed_weights_vector = signs_vector * weights_vector

    return signed_weights_vector

def get_aperiodic_weight_matrix(input_shape, num_units, input_scaling):

    #returns an input matrix, where every row is the weight vector for an input unit and the signs of the weights are created
    #from the decimal expansion of pi, all rows have different signs pattern obtained from creating num_unit long 
    #expansions of pi 

    ta = tf.TensorArray(dtype=tf.float32, size=input_shape)

    start_digit = 0

    for i in range(input_shape):
        signed_weights_vector = get_signed_vector(num_units,start_digit=start_digit, scaling= input_scaling)

        ta = ta.write(i,signed_weights_vector)
        start_digit += num_units

    matrix = ta.stack()
    
    return matrix

# ------------------------------------- LAYERS --------------------------------------------------

# Simple Cycle Cell - from Claudio Gallicchio's EulerESN Github
class RingReservoirCell(keras.layers.Layer):
#builds a ring reservoir as a hidden dynamical layer for a recurrent neural network
#differently from a conventional reservoir layer, in this case the units in the recurrent
#layer are organized to form a cycle (i.e., a ring),


    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.activation = activation
        self.count = 0
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
              
        #build the recurrent weight matrix
        I = tf.linalg.eye(self.units)
        W = self.spectral_radius * tf.concat([I[:,-1:],I[:,0:-1]],axis = 1)
        self.recurrent_kernel = W               
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
        
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True



    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        self.count += 1

        return output, [output]
    
class MFRingReservoirCell(keras.layers.Layer):
    #Memristive - Friendly implementation of the Ring Layer

    def __init__(self,  
                 units, dt=1, kd0=0.5, etap=10, etad=1, alpha=1.0,
                 memory_factor = 0.9, 
                 kp0= 0.0001, 
                 activation = tf.nn.tanh,
                input_scaling = 1.0, bias_scaling = 1.0, gamma = 1.0, p = 1.0,
                 **kwargs):
        
        self.units = units 
        self.state_size = units
        self.memory_factor = memory_factor  # desired memory factor (spectral radius)
        self.activation = activation

        self.dt = dt
        self.kp0 = kp0
        self.kd0 = kd0
        self.etap = etap
        self.etad = etad
        self.alpha = alpha
        self.gamma = gamma
        self.p = p

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        I = tf.linalg.eye(self.units)
        W = self.memory_factor * tf.concat([I[:,-1:],I[:,0:-1]],axis = 1)
        self.recurrent_kernel = W               
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
        
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel) 

        z = (input_part + state_part + self.bias)
        z = scale_z(z, self.p)
        q = (self.kp0*e**(self.etap*z*self.alpha) + self.kd0*e**(-self.etad*z*self.alpha)) * prev_output
        r = self.kp0*e**(self.etap*z*self.alpha) - q
        output = self.dt * r + self.gamma*prev_output
        
        return output, [output]

#ESN and MF-ESN cells - from Veronica Pistoles's Github
class MFReservoirCell(keras.layers.Layer):
    # Reservoir Cell of Memristive-Friendly ESN (MF-ESN)

    def __init__(self,  
                 units, dt=1, kd0=0.5, etap=10, etad=1, alpha=1.0,
                 memory_factor = 0.9, 
                 kp0= 0.0001, 
                 activation = tf.nn.tanh,
                input_scaling = 1.0, bias_scaling = 1.0, gamma = 1.0, p = 1.0,
                 **kwargs):
        
        self.units = units 
        self.state_size = units
        self.memory_factor = memory_factor  # desired memory factor (spectral radius)
        self.activation = activation

        self.dt = dt
        self.kp0 = kp0
        self.kd0 = kd0
        self.etap = etap
        self.etad = etad
        self.alpha = alpha
        self.gamma = gamma
        self.p = p

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        super().__init__(**kwargs)
        
    def build(self, input_shape):

        #build the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value  = (self.memory_factor / np.sqrt(self.units)) * (6/np.sqrt(12))
        W = tf.random.uniform(shape = (self.units, self.units), minval = -value,maxval = value)
        self.recurrent_kernel = W   
        #print('input_shape', input_shape)
        #input("Press Enter to continue...")
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
        #initialize the bias 
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)

        self.built = True


    def call(self, inputs, states):
        
        prev_output = states[0]
        #print(prev_output)
        #input("Press Enter to continue...")
        
        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel) 

        z = (input_part + state_part + self.bias)
        z = scale_z(z, self.p)
        q = (self.kp0*e**(self.etap*z*self.alpha) + self.kd0*e**(-self.etad*z*self.alpha)) * prev_output
        r = self.kp0*e**(self.etap*z*self.alpha) - q
        output = self.dt * r + self.gamma*prev_output
        
        return output, [output]

class ReservoirCell(keras.layers.Layer):
    #builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units 
        self.state_size = units
        self.input_scaling = input_scaling 
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky #leaking rate
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value  = (self.spectral_radius / np.sqrt(self.units)) * (6/np.sqrt(12))
        W = tf.random.uniform(shape = (self.units, self.units), minval = -value,maxval = value)
        self.recurrent_kernel = W   
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
                         
        #initialize the bias 
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):

        prev_output = states[0]
        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        
        return output, [output]

#MinCompESN and MF-MinCompESN cells
class RingReservoirCell_MinComp(keras.layers.Layer):
#version of the Cell using UDAP from the paper "Minimum Complexity Echo State Network" from Rodan et. al.
#the input weights are set to be all the same value = input_scaling
#and the signs follow a Universal Deterministic Aperiodic Pattern created from the digits of pi

    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
              
        #build the recurrent weight matrix
        I = tf.linalg.eye(self.units)
        W = self.spectral_radius * tf.concat([I[:,-1:],I[:,0:-1]],axis = 1)
        self.recurrent_kernel = W               
        
        #build the input weight matrix
        self.kernel = get_aperiodic_weight_matrix(input_shape=input_shape[-1], num_units= self.units, input_scaling= self.input_scaling)

        self.bias = get_signed_vector(self.units,0,self.bias_scaling) #the sign sequence for the bias always starts from the beginning of the digits of pi
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky

        return output, [output]

class MFRingReservoirCell_MinComp(keras.layers.Layer):
    #Memristive - Friendly implementation of the Ring Layer

    def __init__(self,  
                 units, dt=1, kd0=0.5, etap=10, etad=1, alpha=1.0,
                 memory_factor = 0.9, 
                 kp0= 0.0001, 
                 activation = tf.nn.tanh,
                input_scaling = 1.0, bias_scaling = 1.0, gamma = 1.0, p = 1.0,
                 **kwargs):
        
        self.units = units 
        self.state_size = units
        self.memory_factor = memory_factor  # desired memory factor (spectral radius)
        self.activation = activation

        self.dt = dt
        self.kp0 = kp0
        self.kd0 = kd0
        self.etap = etap
        self.etad = etad
        self.alpha = alpha
        self.gamma = gamma
        self.p = p

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        I = tf.linalg.eye(self.units)
        W = self.memory_factor * tf.concat([I[:,-1:],I[:,0:-1]],axis = 1)
        self.recurrent_kernel = W               
        
        #build the input weight matrix
        self.kernel = get_aperiodic_weight_matrix(input_shape=input_shape[-1], num_units= self.units, input_scaling= self.input_scaling)
        
        self.bias = get_signed_vector(self.units,0,self.bias_scaling) #the sign sequence for the bias always starts from the beginning of the digits of pi

        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel) 

        z = (input_part + state_part + self.bias)
        z = scale_z(z, self.p)
        q = (self.kp0*e**(self.etap*z*self.alpha) + self.kd0*e**(-self.etad*z*self.alpha)) * prev_output
        r = self.kp0*e**(self.etap*z*self.alpha) - q
        output = self.dt * r + self.gamma*prev_output
        
        return output, [output]

# ----------------------------------- MODELS -------------------------------------------------

class RingESN(keras.Model):
    #Implements a Ring-Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification

    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = RingReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)

class RingESNregr(keras.Model):
    #Implements a Ring-Echo State Network model for time-series regression problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification

    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = RingReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse
    
class MFRingESN(keras.Model):
    #Memristive Friendly implementation of the Ring-Echo State Network, for Classification

    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFRingReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)
    
class MFRingESNregr(keras.Model):
    #Memristive Friendly implementation of the Ring-Echo State Network, for Regression

    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFRingReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse

class MF(keras.Model):
    # Implements a Memristive-Friendly ESN (MF-ESN) model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with MFReservoirCell,
    # followed by a trainable readout layer for classification
    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

    
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)

class MFregr(keras.Model):
    # Implements a Memristive-Friendly ESN model (MF-ESN) for time-series regression problems
    #
    # The architecture comprises a recurrent layer with MFReservoirCell,
    # followed by a trainable readout layer for regression
    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        #x_train_states è l'esecuzione del reservoir sugli input, dato che i valori dei pesi sono fissati, viene calcolato solo una volta
        #rappresenta i valori che escono dal reservoir per ogni input della serie,
        #essendo fissi, si calcolano solo una volta e si allena (fit) solo il layer alla fine

        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse

class ESN(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units,
                 input_scaling = 1.0, bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')


        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        #print('x_train_states', x_train_states)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)
 
class ESNregr(keras.Model):
    # Implements an Echo State Network model for time-series regression problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for regression
    
    def __init__(self, units,
                 input_scaling=1.0, bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1, 
                 readout_regularizer=1.0,
                 activation=tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
            keras.layers.RNN(cell=ReservoirCell(units=units,
                                                input_scaling=input_scaling,
                                                bias_scaling=bias_scaling,
                                                spectral_radius=spectral_radius,
                                                leaky=leaky))
        ])
        self.readout = Ridge(alpha=readout_regularizer, solver='svd')

    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output

    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse


#MinComp models
class RingESN_MinComp(keras.Model):
    #Implements a Ring-Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification

    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = RingReservoirCell_MinComp(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)

class RingESNregr_MinComp(keras.Model):
    #Implements a Ring-Echo State Network model for time-series regression problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification

    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = RingReservoirCell_MinComp(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse

class MFRingESN_MinComp(keras.Model):
    #Memristive Friendly implementation of the Ring-Echo State Network, for Classification

    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFRingReservoirCell_MinComp(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)
    
class MFRingESNregr_MinComp(keras.Model):
    #Memristive Friendly implementation of the Ring-Echo State Network, for Regression

    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFRingReservoirCell_MinComp(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse




