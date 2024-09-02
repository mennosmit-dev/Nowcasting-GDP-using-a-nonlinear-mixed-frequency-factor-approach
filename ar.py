'''This module contains the main program for nowcasting GDP using AR(1).'''


import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.linear_model import Ridge

# Initial Data Settings
first_month_in_data = 241   # Is the first month in the training data, m = 241 corresponds to 1980-01, adjust accordingly
start_month = 601  # Is the first out of sample vintage monthly data set, m = 601 corresponds to 2010-01, adjust accordingly
end_month = 772   # Is the last out of sample vintage monthly data set, m = 772 corresponds to 2024-04
OOS_obs = end_month - start_month  # Number of out-of-sample observations
use_vintage_data = True  # Bool to indicate whether vintage data is used, or not (meaning only the last vintage data)

# Getting the last transformed quarterly vintage data needed to calculate error later
prepare_current_data(end_month, first_month_in_data, use_vintage_data)
last_vintage_data_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)

# The current month is initialised
current_month = start_month

# Initialisation of array to AR(1) errors
e_pred1 = np.zeros([OOS_obs, 1])

# For all months in the out-of-sample an estimation is made
for t in range(OOS_obs):

        # The current month is updated
        current_month = start_month + t

        # For the current month the data is extracted and prepared for use
        prepare_current_data(current_month, first_month_in_data, use_vintage_data)
        current_data_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)

        # The value for next quarters GDP is accessed
        quarters_to_go = last_vintage_data_quarterly.shape[0] - current_data_quarterly.shape[0]
        actual_data = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - quarters_to_go]  # Access the actual transformed GDP we try to predict

        # The last non-zero element is acces for AR(1) prediction
        last_element = None
        for i in range(1, len(current_data_quarterly) + 1):
            potential_element = current_data_quarterly.iloc[-i, :].dropna() # Removing element if it is zero
            if not potential_element.empty:
                last_element = potential_element
                break
        e_pred1[t] = actual_data - last_element

# Storing predictions errors
e_pred1_df = pd.DataFrame(e_pred1)
e_pred1_df.to_csv('/content/AR(1)_errors.csv', index=False, header=False)
