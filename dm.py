'''This module contains can be used to calculate the DM p-value.'''

import numpy as np
import pandas as pd
from scipy.stats import norm

# Loading in Bridge errors
error_PCA_Bridge = pd.read_csv("/content/PCA_error_Bridge.csv", header=None)
error_SPC_Bridge = pd.read_csv("/content/SPC_error_Bridge.csv", header=None)
error_sigmoid_Bridge = pd.read_csv("/content/KPCsigmoid_error_Bridge.csv", header=None)
error_RBF_Bridge = pd.read_csv("/content/KPCrbf_error_Bridge.csv", header=None)

# Loading in MIDAS-b errors
error_PCA_MIDAS-b = pd.read_csv("/content/PCA_error_MIDAS-b.csv", header=None)
error_SPC_MIDAS-b = pd.read_csv("/content/SPC_error_MIDAS-b.csv", header=None)
error_sigmoid_MIDAS-b = pd.read_csv("/content/KPCsigmoid_error_MIDAS-b.csv", header=None)
error_RBF_MIDAS-b = pd.read_csv("/content/KPCrbf_error_MIDAS-b.csv", header=None)

# Loading in MIDAS-a errors
error_PCA_MIDAS-e = pd.read_csv("/content/PCA_error_MIDAS-e.csv", header=None)
error_SPC_MIDAS-e = pd.read_csv("/content/SPC_error_MIDAS-e.csv", header=None)
error_sigmoid_MIDAS-e = pd.read_csv("/content/KPCsigmoid_error_MIDAS-e.csv", header=None)
error_RBF_MIDAS-e = pd.read_csv("/content/KPCrbf_error_MIDAS-e.csv", header=None)

# Selecting the correct erros: full period '.values', first months '.values[::3]', second months '.values[1::3]', third months '.values[2::3]', covid '.values[120:132]', excluding covid '.drop( .index[120:132]).values' (fill in error)
# Select the erros to compare
e1 = error_PCA_Bridge.values
e2 = error_SPC_Bridge.values

DM_statistic = dmtest(e1,e2,1)
p_value = 2 * (1 - norm.cdf(abs(DM_statistic)))
print(p_value)
