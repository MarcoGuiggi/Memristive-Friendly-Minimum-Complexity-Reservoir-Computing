import numpy as np
from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt

# === CLASSIFICATION TASK ===

datasets_class = [
    "Coffee", "Wafer", "SyntheticControl", "ArticularyWordRecognition", 
    "JapaneseVowels", "ItalyPowerDemand", "GunPoint", "ECG5000"
    , "EMOPain", "FordA", "ChlorineConcentration", "Worms"
]

models_class = ["ESN", "MF-ESN", "RingESN", "MF-RingESN", "MinCompESN", "MF-MinCompESN"]

accuracy_values = np.array([
    [0.764, 0.986, 1.000, 1.000, 0.643, 0.571],
    [0.995, 0.991, 0.995, 0.988, 0.992, 0.955],
    [0.983, 0.991, 0.993, 0.985, 0.987, 0.983],
    [0.970, 0.936, 0.967, 0.916, 0.947, 0.903],
    [0.986, 0.991, 0.990, 0.989, 0.989, 0.995],
    [0.870, 0.918, 0.871, 0.907, 0.882, 0.857],
    [0.919, 0.855, 0.931, 0.959, 0.540, 0.547],
    [0.924, 0.930, 0.930, 0.931, 0.930, 0.938],
    [0.686, 0.703, 0.641, 0.761, 0.741, 0.766],
    [0.650, 0.654, 0.666, 0.535, 0.501, 0.513],
    [0.616, 0.587, 0.658, 0.587, 0.646, 0.557],
    [0.308, 0.355, 0.307, 0.330, 0.338, 0.338]
])


fig1, ax1 = plot_critical_difference(accuracy_values, models_class)
plt.title("Critical Difference - Classification")
plt.show()


