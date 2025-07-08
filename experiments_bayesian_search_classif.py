#experiments on ESN, MF-ESN, Ring, MF-Ring with bayesian search for moodel selection

from datetime import datetime
from Memristive_Friendly_Minimum_Complexity_ESN import * 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras # type: ignore
import keras_tuner
import os
import random
from time import time, localtime, strftime
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
import itertools
from Bayesian_Search_Space import *
from Bayesian_Search_Hypermodels import *

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

root_path = "./Bayesian Search Best Results"
max_trials = 1 #number of different combinations to try
executions_per_trial = 1 #number of trainings and evaluations to average on the same combination of params
num_guesses = 1 #numbers of trials for the final test

# Hypermodels and Search Space configuration parameters are in their respective separate python code
  
#-----------------------CLASSIFICATION experiments------------------------

time_string_start = strftime("%H.%M.%S", localtime())
last_results_path = os.path.join(root_path)
if not os.path.exists(last_results_path):
        os.makedirs(last_results_path)
last_results_filename = os.path.join(last_results_path, 'Last Experiment Best Results Comparison classification.txt')
last_results_logger = open(last_results_filename, 'w')



#datasets = ['JapaneseVowels','SyntheticControl','ECG5000', 'Epilepsy','Coffee', 'Wafer', 'GunPoint','EMOPain','FordA','ArticularyWordRecognition','ChlorineConcentration','ItalyPowerDemand','OliveOil','Worms']
datasets = ['EMOPain','FordA','ArticularyWordRecognition','ChlorineConcentration',
            'ItalyPowerDemand','OliveOil','Worms']

time_string_start = strftime("%H.%M.%S", localtime())
results_comparison_path = os.path.join(root_path, 'Results Comparison')
if not os.path.exists(results_comparison_path):
    os.makedirs(results_comparison_path)
results_comparison_logger_filename = os.path.join(results_comparison_path, 'Best Results Comparison_classification'+time_string_start+ '.txt')
results_comparison_logger = open(results_comparison_logger_filename, 'w')

for dataset_name in datasets:
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_comparison_logger.write('\n\n##################################################################################\n')
    results_comparison_logger.write('Model selection through Bayesian Search on dataset ' + dataset_name + '\n')
    results_comparison_logger.write(f"Number of parameters combinations per model type: {max_trials}, executions per params combination: {executions_per_trial}\n")
    results_comparison_logger.write(f"\ttuning readout regularizer in the bayesian search\n")

    last_results_logger.write('\n\n##################################################################################\n')
    last_results_logger.write('Model selection through Bayesian Search on dataset ' + dataset_name + '\n')
    last_results_logger.write(f"Number of parameters combinations per model type: {max_trials}, executions per params combination: {executions_per_trial}\n")
    last_results_logger.write(f"\ttuning readout regularizer in the bayesian search\n")

    #for model_type in ["ESN","MF-ESN","Ring","MF-Ring"]:
    for model_type in ["Ring_MinComp", "MF-Ring_MinComp"]:

        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        x_train_all_aeon, y_train_all_aeon = load_classification(name=dataset_name, split="train")
        x_test_aeon, y_test_aeon = load_classification(name=dataset_name, split="test")

        x_train_all = np.transpose(x_train_all_aeon, (0, 2, 1))
        x_test = np.transpose(x_test_aeon, (0, 2, 1))
        _, y_train_all = np.unique(y_train_all_aeon, return_inverse=True)
        _, y_test = np.unique(y_test_aeon, return_inverse=True)

        x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all)

        initial_model_selection_time = time()
        hypermodel = None
        if(model_type == "ESN"):
            hypermodel = HyperESN_classification()
        elif(model_type == "Ring"):
            hypermodel = HyperRing_classification()
        elif(model_type == "MF-ESN"):
            hypermodel = HyperMF_ESN_classification()
        elif(model_type == "MF-Ring"):
            hypermodel = HyperMF_Ring_classification()
        elif(model_type == "Ring_MinComp"):
             hypermodel = HyperRing_MinComp_classification()
        elif(model_type == "MF-Ring_MinComp"):
             hypermodel = HyperMF_Ring_MinComp_classification()

        
        now = datetime.now()
        time_str = now.strftime("%H.%M.%S")
        date_str = now.strftime("%d_%m_%y")

        tuner = keras_tuner.BayesianOptimization(
                hypermodel,
                objective="val_accuracy", #we aim to find the parameters that yield the best validation accuracy
                max_trials= max_trials,
                executions_per_trial=executions_per_trial,
                directory = "oracle BayesianSearch logs", #directory for the really non user friendly oracle logs, cannot be disabled so they go in a separate folder
                project_name = model_type+"/"+model_type+"_results_log at "+time_str+" on "+date_str,
                overwrite = True #wether to overwrite the project logs on a new execution or to continue with the new data on the same directory
        )

        search_start = datetime.now().strftime("%H:%M:%S")
        print("\n\n\n\n\n\n\n-----------------------------------------------------------")
        print(f"[{search_start}] Now Tuning {model_type} on {dataset_name}...")

        tuner.search(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    verbose=1 #specifies thoroughness of runtime console logs
                )

        elapsed_model_selection_time = time() - initial_model_selection_time

        best_params = tuner.get_best_hyperparameters(1)[0]

        search_end = datetime.now().strftime("%H:%M:%S")

        tuner.results_summary(1)


        if best_params:
            if(model_type == "ESN"):
                hypermodel = HyperESN_classification()
            elif(model_type == "Ring"):
                hypermodel = HyperRing_classification()
            elif(model_type == "MF-ESN"):
                hypermodel = HyperMF_ESN_classification()
            elif(model_type == "MF-Ring"):
                hypermodel = HyperMF_Ring_classification()
            elif(model_type == "Ring_MinComp"):
                hypermodel = HyperRing_MinComp_classification()
            elif(model_type == "MF-Ring_MinComp"):
                hypermodel = HyperMF_Ring_MinComp_classification()
            else:
                print(f"\n\n!!!!{model_type} not implemented for final testing!!\n\n")
                break

            acc_ts = []
            required_time = []
            initial_time = time()

            for i in range(executions_per_trial):

                best_model = hypermodel.build(best_params)
                best_model.fit(x_train_all, y_train_all)
                acc = best_model.evaluate(x_test, y_test)
                required_time.append(time() - initial_time)
                acc_ts.append(acc)

            #------------------Logging------------------
            
            time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())

            print(f'--{model_type} on {dataset_name}--')
            print(f'Results: MEAN {np.mean(acc_ts)} STD {np.std(acc_ts)}')
            print(f'Required time: MEAN {np.mean(required_time)} STD {np.std(required_time)}')
            print(f'Total model selection time: {elapsed_model_selection_time}')

            #comparison logger
            results_comparison_logger.write("\n---------------------------------------------------------\n")
            results_comparison_logger.write(f"{model_type} started at {time_string_start}, ended at {time_string_end}\n")
            results_comparison_logger.write(f'\nModel selection time: {elapsed_model_selection_time} seconds = {elapsed_model_selection_time / 60} minutes\n')
            results_comparison_logger.write(f'Accuracy: \n\tMEAN {np.mean(acc_ts)} \n\tSTD {np.std(acc_ts)}\n\n')
            results_comparison_logger.write(f'Best parameters: \n')
            for name, value in best_params.values.items():
                results_comparison_logger.write(f"{name}: {value}\n")

            last_results_logger.write("\n---------------------------------------------------------\n")
            last_results_logger.write(f"{model_type} started at {time_string_start}, ended at {time_string_end}\n")
            last_results_logger.write(f'\nModel selection time: {elapsed_model_selection_time} seconds = {elapsed_model_selection_time / 60} minutes\n')
            last_results_logger.write(f'Accuracy: \n\tMEAN {np.mean(acc_ts)} \n\tSTD {np.std(acc_ts)}\n\n')
            last_results_logger.write(f'Best parameters: \n')
            for name, value in best_params.values.items():
                        last_results_logger.write(f"{name}: {value}\n")
        else:
            print(f'No valid model found during model selection for {model_type}.')
            results_comparison_logger.write(f'No valid model found during model selection for {model_type}.\n')
            last_results_logger.write(f'No valid model found during model selection for {model_type}.\n')
            last_results_logger.close()
            results_comparison_logger.close()




results_comparison_logger.close()
last_results_logger.close()

