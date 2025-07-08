import numpy as np
from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt

# === REGRESSION TASK ===

datasets_reg = [
    "FloodModeling1", "FloodModeling2", "FloodModeling3",
    "Covid3Month", "IEEEPPG", "AppliancesEnergy", "NewsHeadlineSentiment"
]

models_reg = ["ESN", "MF-ESN", "RingESN", "MF-RingESN", "MinCompESN", "MF-MinCompESN"]

error_means = np.array([
    [0.00828083, 0.01622491, 0.00808105, 0.01858117, 0.0180, 0.0189],
    [0.01226061, 0.01851190, 0.01186991, 0.01852247, 0.0180, 0.0185],
    [0.00904332, 0.01942355, 0.00888306, 0.02271747, 0.0208, 0.0227],
    [0.04265337, 0.04250023, 0.04252379, 0.04235868, 0.0438, 0.0446],
    [35.34796920, 32.89181825, 35.04340282, 33.29263750, 36.4801, 32.8082],
    [3.15679208, 3.45532611, 3.29564448, 3.46525019, 3.4549, 3.4549],
    [0.14223459, 0.14225157, 0.14223705, 0.14225030, 0.1422, 0.1423]
])

fig2, ax2 = plot_critical_difference(error_means, models_reg, lower_better= True)
plt.title("Critical Difference - Regression")
plt.show()
