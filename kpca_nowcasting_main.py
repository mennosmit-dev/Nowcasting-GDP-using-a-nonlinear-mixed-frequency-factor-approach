'''This module contains the main program for nowcasting GDP using kPCA.'''


!pip install rpy2 #installs R in python
!pip install xlsxwriter #installs functionality to read and write .xlsx documents
!pip install pandas openpyxl #needed to open xlsx files in python
!pip install scipy #installs NLS for MIDAS needed
!pip install git+https://github.com/mikemull/midaspy.git #installs part of MIDAS that we use

# Importing functionalities from different libraries
from sklearn.linear_model import Ridge
from midas.weights import polynomial_weights, WeightMethod, BetaWeights, ExpAlmonWeights
from additional_functions import lagaugment, dmtest, getPCs, prepare_current_data, transformation_function, Setup_Imputation, Impute, estimate_general, forecast_general, new_x_weighted, ssr_generalized, jacobian_generalized
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm

# Replacing x_weighted function from midaspy by adopted version defined in additional_functions
BetaWeights.x_weighted = new_x_weighted
ExpAlmonWeights.x_weighted = new_x_weighted

# Initial Data Settings
first_month_in_data = 241   # Is the first month in the training data, m = 241 corresponds to 1980-01, adjust accordingly
start_month = 601  # Is the first out of sample vintage monthly data set, m = 601 corresponds to 2010-01, adjust accordingly
end_month = 772   # Is the last out of sample vintage monthly data set, m = 772 corresponds to 2024-04
OOS_obs = end_month - start_month  # Number of out-of-sample observations
use_vintage_data = True  # Bool to indicate whether vintage data is used, or not (meaning only the last vintage data)

# Initialisation of maximum number of model components
P = 1  # Maximum number of lags of Y (2 is equivalent to one lag maximum)
K = 6  # Maximum number of PC factors
H = 6  # Maximum number of PC factor lags

# The choice for the principal components, choices are ['PCA'], ['SPC'], ['KPCsigmoid'], and ['KPCrbf']
Modes = ['KPCsigmoid']

# Scaling factors for BIC of Bridge and MIDAS respectively, beta = 1 corresponds to standard BIC
beta1 = 1
beta2 = 1

# Ridge penalty, lambda_1 = 1 corresponds to OLS
lambda_1 = 0

# Getting the last transformed quarterly vintage data needed to calculate error later
prepare_current_data(end_month, first_month_in_data, use_vintage_data)
last_vintage_data_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)

# The current month is initialised
current_month = start_month

# A reference value is used to later calculate how many rows need to be imputed in a month
referenceValue = 0

# For all selected PCA methods the estimation is done
for Mode in Modes:  # For all PC methods we do the training and prediction part at each time
    print("Mode " + Mode + " started!" )

    # The grid corresponding to the PC method is initialised
    if Mode[0] == 'K':
        kernel = Mode[3:]  # extracts kernel type
    else:
        kernel = False
    if kernel == 'sigmoid':
        grid = 10**np.arange(-6, -1.5, 0.5)  # -6,-1.5,.5 with 1 at the end 5 grid parameters, with 0.5 at the end 10 grid parameters
    elif kernel == 'rbf':
        grid = 10**np.arange(-6, -2.5, 0.5)  # -6,-2.5,.5
    else:
        grid = [1]

    # Initialisation of arrays to store predictions, errors, and optimal parameters for Bridge, MIDAS-b and MIDAS-e
    Y_pred1 = np.zeros([OOS_obs, 1])
    e_pred1 = np.zeros([OOS_obs, 1])
    optimal_params1 = np.empty(OOS_obs, dtype=object)
    Y_pred2 = np.zeros([OOS_obs, 1])
    e_pred2 = np.zeros([OOS_obs, 1])
    optimal_params2 = np.empty(OOS_obs, dtype=object)
    Y_pred3 = np.zeros([OOS_obs, 1])
    e_pred3 = np.zeros([OOS_obs, 1])
    optimal_params3 = np.empty(OOS_obs, dtype=object)

    # For all months in the out-of-sample an estimation is made based on the optimal model, and the corresponding error is calculated
    for t in range(OOS_obs):

        # The current month is updated
        current_month = start_month + t

        # For the current month the data is extracted and prepared for use
        prepare_current_data(current_month, first_month_in_data, use_vintage_data)
        current_data_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)
        current_data_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)

        # The value for next quarters GDP is accessed
        quarters_to_go = last_vintage_data_quarterly.shape[0] - current_data_quarterly.shape[0]
        actual_data = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - quarters_to_go]

        # During the first month the referenceValue is set which is used to determine how many rows to add after
        if current_month == start_month:
            referenceValue = current_data_monthly.shape[0]

        # The number of months that needs to be imputed via EM algorithm is calculated
        added_rows = 0
        if (current_data_monthly.shape[0] - referenceValue) % 3 == 0: #in this case month 1, 4, 7, 10
            added_rows = 3
        elif (current_data_monthly.shape[0] - referenceValue) % 3 == 1: #in this case month  2, 5, 8, 11
            added_rows = 2
        else: #in this case month 1, 4, 7, 10; we generate one month with EM extra  3, 6, 9 and 12
            added_rows = 1

        # NaN rows are added and imputed via EM algorithm
        for i in range(added_rows):
            Nan_row = pd.DataFrame([[np.nan] * current_data_monthly.shape[1]], columns=current_data_monthly.columns) # NaN row generated
            current_data_monthly = pd.concat([current_data_monthly, Nan_row], ignore_index=True) # NaN row concatenated
            random_cols = random.sample(list(current_data_monthly.columns), 13) # 13 variables are randomly chosen to be imputed via AR(1)
            for col in random_cols:  # The imputation is needed for the EM algorithm to work, as more elements need to be known than max factors, which we chose as 12
                current_data_monthly.iloc[-1, current_data_monthly.columns.get_loc(col)] = current_data_monthly.iloc[-2, current_data_monthly.columns.get_loc(col)] # AR(1) imputation
            current_data_monthly.to_csv("/content/Monthly_Data_Modified.csv", header = None, index=False)
            Impute() # rest of variables is imputed based on EM algorithm
            current_data_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)

        # For Bridge, MIDAS-b, and MIDAS-e the resulting optimal MSPE is stored for each hyperparameter
        BigTable1 = {}
        BigTable2 = {}
        BigTable3 = {}

        # For each grid value the optimal model is found for the training data and is used to forecast in the last two comparable months
        for hypernum, hyper in enumerate(grid):
            print("hypernum")
            print(hypernum)

            # The principal components are extracted
            Fhat = getPCs(current_data_monthly, Mode, hyper, K) #extract the maximum number of PCs for the current method and grid

            # the BIC values are stored for each combination of factors, lags of factors, and lags of GDP
            BIC1 = np.zeros([K, H, P+1])
            BIC2 = np.zeros([K, H, P+1])
            BIC3 = np.zeros([K, H, P+1])

            # For each combination of no lag of y or one lag
            for pp in range(P+1):

                if pp == 0:  # No lag
                    Y = current_data_quarterly.values
                    Y = Y[1:]   # Removes first quarter, is needed as Ylag's first value is NaN
                    Ylag = None
                else: # One lag
                    [Y, Ylag] = lagaugment(current_data_quarterly.values, 1)  # Lag of Y is formed
                    Y = Y[1:]  # Removes first quarter, is needed as Ylag's first value is NaN
                    Ylag = Ylag[1:] # Ylag's first value is NaN and must be deleted

                # For each combination of up to K principal components
                for kk in range(K):
                    Fhat_selected_original = Fhat[:, :kk+1]

                    # For each combination of up to H lags of the principal components
                    for mm in range(H):
                        nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan)

                        # NaN row added to bottom to compensate for lagaument function
                        Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

                        [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mm + 1) # The lags of de PCs are calculated and subsequently combined into one matrix
                        Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kk + 1), Fhat_selectedLag[:, 0:-(kk + 1)]))

                        # The NaN values are filled up which is needed for NLS convergence within MIDAS
                        Fhat_selected = pd.DataFrame(Fhat_selected)
                        Fhat_selected.fillna(Fhat_selected.median(), inplace=True) # Imputing the median of the lag for each NaN
                        Fhat_selected = Fhat_selected.to_numpy()


                        # Bridge estimation --------

                        # Averaging data across quarters
                        number_of_quarters = Fhat_selected.shape[0] // 3
                        reshaped_data = Fhat_selected.reshape(number_of_quarters, 3, Fhat_selected.shape[1])
                        Fhat_averaged = np.nanmean(reshaped_data, axis=1)  # Computes the mean along the second axis, which is the months

                        # Setting up the regressor data
                        if Ylag is None:
                            X_historical = Fhat_averaged[1:-1]  # The first quarter is excluded similarly to Y and Ylag, the last one is used for the prediction at the end
                        else:
                            X_historical = np.hstack((Fhat_averaged[1:-1], Ylag))  # The first quarter is excluded similarly to Y and Ylag, the last one is used for the prediction at the end
                        X_historical = sm.add_constant(X_historical) # adding a one vector as constant to the regressor set

                        # OLS/RIDGE estimation, lambda_1 = 0 is OLS, lambda_1 > 0 is Ridge
                        model = Ridge(alpha=lambda_1)
                        model.fit(X_historical, Y)  # Estimates the model
                        Y_pred = model.predict(X_historical)  # Makes predictions for the whole in sample period
                        Y_pred = Y_pred.reshape(Y_pred.shape[0], 1) # Needed to subtract from Y in sample_var

                        # Calculating number of observations and number of variables needed for sample_var
                        Tnum = X_historical.shape[0] # Number of observations
                        Vnum = X_historical.shape[1] # Number of variables

                        # The sample_var is calculated, is then used to calculate BIC for the combination of regressors
                        sample_var = ((Y - Y_pred)**2) / (Tnum - Vnum)
                        MSPE = np.dot(sample_var.T, sample_var).item()
                        BIC1[kk, mm, pp] = np.log(MSPE) + beta1*(Vnum) * np.log(Tnum) / Tnum


                        # MIDAS data setup ------------

                        X_historical = Fhat_selected[:-3, :]  # We take everything but the last three months which are not for estimation

                        # MIDAS functions require special formatting of regressors which is done here
                        number_of_regressors = X_historical.shape[1]
                        months = 3
                        reshaped_X_historical = np.zeros((X_historical.shape[0] // months, number_of_regressors*months))
                        for j in range(number_of_regressors):
                            current_start = 0
                            for i in range(X_historical.shape[0]):
                                current_row = i // months
                                current_addition = i % months
                                reshaped_X_historical[current_row, j*months + current_addition] = X_historical[i, j]


                        # MIDAS-b estimation -----------

                        # For each regressor a polynomial
                        polys = ['beta'] * (X_historical.shape[1])

                        # Estimating MIDAS-b model and making predictions in training sample
                        if Ylag is None:
                            result = estimate_general(Y, None, reshaped_X_historical[1:], polys, lambda_1)
                            Y_pred = forecast_general(reshaped_X_historical[1:], None, result, polys)
                            Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
                        else:
                            result = estimate_general(Y, Ylag, reshaped_X_historical[1:], polys, lambda_1)
                            Y_pred = forecast_general(reshaped_X_historical[1:], Ylag, result, polys)
                            Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)

                        # The sample_var is calculated, is then used to calculate BIC for the combination of regressors
                        sample_var = ((Y - Y_pred)**2) / (Tnum - Vnum)
                        MSPE = np.dot(sample_var.T, sample_var).item()
                        BIC2[kk, mm, pp] = np.log(MSPE) + beta2*(Vnum) * np.log(Tnum) / Tnum


                        # MIDAS-e estimation -----------

                        # For each regressor a polynomial
                        polys = ['expalmon'] * (X_historical.shape[1])

                        # Estimating MIDAS-b model and making predictions in training sample
                        if Ylag is None:
                            result = estimate_general(Y, None, reshaped_X_historical[1:], polys, lambda_1)
                            Y_pred = forecast_general(reshaped_X_historical[1:], None, result, polys)
                            Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
                        else:
                            result = estimate_general(Y, Ylag, reshaped_X_historical[1:], polys, lambda_1)
                            Y_pred = forecast_general(reshaped_X_historical[1:], Ylag, result, polys)
                            Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)

                        # The sample_var is calculated, is then used to calculate BIC for the combination of regressors
                        sample_var = ((Y - Y_pred)**2) / (Tnum - Vnum)
                        MSPE = np.dot(sample_var.T, sample_var).item()
                        BIC3[kk, mm, pp] = np.log(MSPE) + beta2*(Vnum) * np.log(Tnum) / Tnum

                        #-----------------


            # For the current parameter, within the three different regressions, the optimal combination of regressors is retrieved
            [kopt1, mopt1, popt1] = [np.where(BIC1 == np.min(BIC1))[0][0] + 1, np.where(BIC1 == np.min(BIC1))[1][0] + 1, np.where(BIC1 == np.min(BIC1))[2][0]]
            [kopt2, mopt2, popt2] = [np.where(BIC2 == np.min(BIC2))[0][0] + 1, np.where(BIC2 == np.min(BIC2))[1][0] + 1, np.where(BIC2 == np.min(BIC2))[2][0]]
            [kopt3, mopt3, popt3] = [np.where(BIC3 == np.min(BIC3))[0][0] + 1, np.where(BIC3 == np.min(BIC3))[1][0] + 1, np.where(BIC3 == np.min(BIC3))[2][0]]


            # The optimal model is used to make predictions for the last two comparable months --------------------start

            ct = 4  # The number of data_sets that are kept in storage simulatenously (do not adjust)

            # Initialisation of the predictions made per regression method for validation (but only the first and fourth elements are used for the two comparable months)
            Y_pred1_previous = np.zeros([ct, 1])
            e_pred1_previous = np.zeros([ct, 1])
            Y_pred2_previous = np.zeros([ct, 1])
            e_pred2_previous = np.zeros([ct, 1])
            Y_pred3_previous = np.zeros([ct, 1])
            e_pred3_previous = np.zeros([ct, 1])

            # For all validation observations
            for ll in range(ct): # we use the optimal model for the last ct months, make predictions and get the errors

                # Month is updated (back in time)
                month = current_month -ct -2 +ll

                # During the first time several datesets need to be loaded in, later those are updated in efficient way such that it limits calculation time
                if t == 0 and hypernum == 0 and ll == 0:

                    prepare_current_data(month, first_month_in_data, use_vintage_data)
                    four_months_ago_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)
                    four_months_ago_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)
                    four_months_ego_quarters_to_go = last_vintage_data_quarterly.shape[0] - four_months_ago_quarterly.shape[0]
                    four_months_ego_actual_data_previous = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - four_months_ego_quarters_to_go]

                    prepare_current_data(month+1, first_month_in_data, use_vintage_data)
                    three_months_ago_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)
                    three_months_ago_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)
                    three_months_ego_quarters_to_go = last_vintage_data_quarterly.shape[0] - three_months_ago_quarterly.shape[0]
                    three_months_ego_actual_data_previous = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - three_months_ego_quarters_to_go]

                    prepare_current_data(month +2, first_month_in_data, use_vintage_data)
                    two_months_ago_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)
                    two_months_ago_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)
                    two_months_ego_quarters_to_go = last_vintage_data_quarterly.shape[0] - two_months_ago_quarterly.shape[0]
                    two_months_ego_actual_data_previous = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - two_months_ego_quarters_to_go]

                    prepare_current_data(month+ 3 , first_month_in_data, use_vintage_data)
                    one_months_ago_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)
                    one_months_ago_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)
                    one_months_ego_quarters_to_go = last_vintage_data_quarterly.shape[0] - one_months_ago_quarterly.shape[0]
                    one_months_ego_actual_data_previous = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - one_months_ego_quarters_to_go]

                # Based on the current month the correct data set is extracted
                if ll == 0:
                    current_data_monthly_previous = four_months_ago_monthly
                    current_data_quarterly_previous = four_months_ago_quarterly
                    quarters_to_go_previous = four_months_ego_quarters_to_go
                    actual_data_previous = four_months_ego_actual_data_previous
                elif ll == 1:
                    current_data_monthly_previous = three_months_ago_monthly
                    current_data_quarterly_previous = three_months_ago_quarterly
                    quarters_to_go_previous = three_months_ego_quarters_to_go
                    actual_data_previous = three_months_ego_actual_data_previous
                elif ll == 2:
                    current_data_monthly_previous = two_months_ago_monthly
                    current_data_quarterly_previous = two_months_ago_quarterly
                    quarters_to_go_previous = two_months_ego_quarters_to_go
                    actual_data_previous = two_months_ego_actual_data_previous
                elif ll == 3:
                    current_data_monthly_previous = one_months_ago_monthly
                    current_data_quarterly_previous = one_months_ago_quarterly
                    quarters_to_go_previous = one_months_ego_quarters_to_go
                    actual_data_previous = one_months_ego_actual_data_previous


                # During first time new datasets are used, extra months need to be imputed still (later saved)
                if t == 0 and hypernum == 0:
                    added_rows = 0
                    if month >= start_month:  # Before referenceValue month the calucation is backwards
                        if (current_data_monthly_previous.shape[0] - referenceValue) % 3 == 0:
                            added_rows = 3
                        elif (current_data_monthly_previous.shape[0] - referenceValue) % 3 == 1:
                            added_rows = 2
                        else:
                            added_rows = 1
                    else:  # Normal calucation (similar to training data earlier)
                        if (referenceValue - current_data_monthly_previous.shape[0]) % 3 == 0:
                            added_rows = 3
                        elif (referenceValue - current_data_monthly_previous.shape[0]) % 3 == 1:
                            added_rows = 1
                        else:
                            added_rows = 2

                    # Empty months are added and imputed similar to training data earlier
                    for i in range(added_rows):
                        Nan_row = pd.DataFrame([[np.nan] * current_data_monthly_previous.shape[1]], columns=current_data_monthly_previous.columns)
                        current_data_monthly_previous = pd.concat([current_data_monthly_previous, Nan_row], ignore_index=True)
                        random_cols = random.sample(list(current_data_monthly_previous.columns), 13)
                        for col in random_cols:
                            current_data_monthly_previous.iloc[-1, current_data_monthly_previous.columns.get_loc(col)] = current_data_monthly_previous.iloc[-2, current_data_monthly_previous.columns.get_loc(col)] #AR(1) imputing
                        current_data_monthly_previous.to_csv("/content/Monthly_Data_Modified.csv", header = None, index=False)
                        Impute()
                        current_data_monthly_previous = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)

                    # Data is saved after the extra data has been imputed, for reusibility next month
                    if ll == 0 and t == 0 and hypernum == 0:
                        four_months_ago_monthly = current_data_monthly_previous
                    if ll == 1 and t == 0 and hypernum == 0:
                        three_months_ago_monthly = current_data_monthly_previous
                    if ll == 2 and t == 0 and hypernum == 0:
                        two_months_ago_monthly = current_data_monthly_previous
                    if ll == 3 and t == 0 and hypernum == 0:
                        one_months_ago_monthly = current_data_monthly_previous

                # For 6 months ago and 3 three months ago (comparable quarters) the predictions are made
                if ll == 0 or ll == 3:

                    # Extract principal components
                    Fhat = getPCs(current_data_monthly_previous, Mode, hyper, K)


                    # Bridge estimation --------------
                    if popt1 == 0:
                        Y = current_data_quarterly_previous.values
                        Y = Y[1:] # Removes first quarter, is needed as Ylag's first value is NaN
                        Ylag = None
                    else:
                        [Y, Ylag] = lagaugment(current_data_quarterly_previous.values, 1) # Lag of Y is formed
                        Y = Y[1:] # Removes first quarter, is needed as Ylag's first value is NaN
                        Ylag = Ylag[1:] # Ylag's first value is NaN and must be deleted

                    # Optimal number of PCs extracted
                    Fhat_selected_original = Fhat[:, :kopt1]

                    # NaN row added to bottom to compensate for lagaument function
                    nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan)
                    Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

                    # Optimal number of lags are formed for the PCs
                    [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mopt1)
                    Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kopt1), Fhat_selectedLag[:, 0:-kopt1]))

                    # NaNs from the lags of PC are imputed as is needed for the convergence of NLS
                    Fhat_selected = pd.DataFrame(Fhat_selected)
                    Fhat_selected.fillna(Fhat_selected.median(), inplace=True)  # Imputing for NaNs the median per colom
                    Fhat_selected = Fhat_selected.to_numpy()

                    # Monthly factors averaged to quarterly factors
                    number_of_quarters = Fhat_selected.shape[0] // 3
                    reshaped_data = Fhat_selected.reshape(number_of_quarters, 3, Fhat_selected.shape[1])
                    Fhat_averaged = np.nanmean(reshaped_data, axis=1)

                    # Regressors are set up
                    if Ylag is None:
                        X_historical = Fhat_averaged[1:-1]
                        X_prediction = np.hstack((np.array([[1]]), Fhat_averaged[-1].reshape(1, -1)))
                    else:
                        X_historical = np.hstack((Fhat_averaged[1:-1], Ylag)) #we dont take the last row of Fhat average, we need that for the final prediction
                        X_prediction = np.hstack((np.array([[1]]), Fhat_averaged[-1].reshape(1, -1), Ylag[-1].reshape(1, -1))) #getting correct things for prediction later
                    X_historical = sm.add_constant(X_historical)

                    # OLS/Ridge is estimated and a one-step-ahead forecast is made, lambda_1 = 0 then OLS, lambda_1 > 0 then Ridge
                    model = Ridge(alpha=lambda_1)
                    model.fit(X_historical, Y)
                    Y_pred1_previous[ll] = model.predict(X_prediction)

                    # Error is calculated
                    e_pred1_previous[ll] = actual_data_previous - Y_pred1_previous[ll]


                    #MIDAS-b estimation------------------------

                    if popt2 == 0:
                        Y = current_data_quarterly_previous.values
                        Y = Y[1:] # Removes first quarter, is needed as Ylag's first value is NaN
                        Ylag = None
                    else:
                        [Y, Ylag] = lagaugment(current_data_quarterly_previous.values, 1)  # Lag of Y is formed
                        Y = Y[1:]  # Removes first quarter, is needed as Ylag's first value is NaN
                        Ylag = Ylag[1:]  # Ylag's first value is NaN and must be deleted

                    # Optimal number of PCs extraced
                    Fhat_selected_original = Fhat[:, :kopt2]

                    # NaN row added to bottom to compensate for lagaument function
                    nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan) #generates a NaN row which we add to Fhat_selected, which we need since lagaugment deletes last row.
                    Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

                    # Optimal number of lags are formed for the PCs
                    [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mopt2)
                    Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kopt2), Fhat_selectedLag[:, 0:-kopt2])) #combines the factors and the lags into a single set of factors

                    # NaNs from the lags of PC are imputed as is needed for the convergence of NLS
                    Fhat_selected = pd.DataFrame(Fhat_selected)
                    Fhat_selected.fillna(Fhat_selected.median(), inplace=True)
                    Fhat_selected = Fhat_selected.to_numpy()

                    X_historical = Fhat_selected[:-3, :]  # Everything but the last three months are used for estimation
                    X_prediction = np.hstack( Fhat_selected[-3:].reshape(3, Fhat_selected.shape[1])) # The last three months for prediction

                    # MIDAS functions require special formatting of regressors which is done here
                    number_of_regressors = X_historical.shape[1]
                    months = 3
                    reshaped_X_historical = np.zeros((X_historical.shape[0] // months, number_of_regressors*months))
                    for j in range(number_of_regressors):
                        current_start = 0
                    for i in range(X_historical.shape[0]):
                        current_row = i // months
                        current_addition = i % months
                        reshaped_X_historical[current_row, j*months + current_addition] = X_historical[i, j]

                    # Prediction matrix also put in right format (easier here since less dimensional)
                    X_prediction = Fhat_selected[-3:, :]
                    reshaped_X_prediction = X_prediction.T.flatten()

                    #For each regressor a polynomial
                    polys = ['beta'] * (X_prediction.shape[1])

                    # Estimating MIDAS-b model and making predictions in training sample
                    if Ylag is None:
                        result = estimate_general(Y, None, reshaped_X_historical[1:], polys, lambda_1)
                        Y_pred2_previous[ll] = forecast_general(reshaped_X_prediction, None, result, polys)
                        e_pred2_previous[ll] = actual_data_previous - Y_pred2_previous[ll]
                    else:
                        result = estimate_general(Y, Ylag, reshaped_X_historical[1:], polys, lambda_1)
                        Y_pred2_previous[ll] = forecast_general(reshaped_X_prediction, Ylag[-1].reshape(1, -1), result, polys)
                        e_pred2_previous[ll] = actual_data_previous - Y_pred2_previous[ll]


                    #MIDAS-e estimation------------------------

                    if popt3 == 0:
                        Y = current_data_quarterly_previous.values
                        Y = Y[1:]
                        Ylag = None
                    else:
                        [Y, Ylag] = lagaugment(current_data_quarterly_previous.values, 1) # Lag of Y is formed
                        Y = Y[1:] # Removes first quarter, is needed as Ylag's first value is NaN
                        Ylag = Ylag[1:] # Ylag's first value is NaN and must be deleted

                    # Optimal number of PCs extraced
                    Fhat_selected_original = Fhat[:, :kopt3]

                    # NaN row added to bottom to compensate for lagaument function
                    nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan) #generates a NaN row which we add to Fhat_selected, which we need since lagaugment deletes last row.
                    Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

                    # Optimal number of lags are formed for the PCs
                    [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mopt3)
                    Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kopt3), Fhat_selectedLag[:, 0:-kopt3])) #combines the factors and the lags into a single set of factors

                    # NaNs from the lags of PC are imputed as is needed for the convergence of NLS
                    Fhat_selected = pd.DataFrame(Fhat_selected)
                    Fhat_selected.fillna(Fhat_selected.median(), inplace=True)
                    Fhat_selected = Fhat_selected.to_numpy()

                    X_historical = Fhat_selected[:-3, :]  # Everything but the last three months are used for estimation
                    X_prediction = np.hstack( Fhat_selected[-3:].reshape(3, Fhat_selected.shape[1])) # The last three months for prediction

                    # MIDAS functions require special formatting of regressors which is done here
                    number_of_regressors = X_historical.shape[1]
                    months = 3
                    reshaped_X_historical = np.zeros((X_historical.shape[0] // months, number_of_regressors*months)) #initialisation MIDAS regressor matrix
                    for j in range(number_of_regressors):
                        current_start = 0
                    for i in range(X_historical.shape[0]):
                        current_row = i // months
                        current_addition = i % months
                        reshaped_X_historical[current_row, j*months + current_addition] = X_historical[i, j]

                    # Prediction matrix also put in right format (easier here since less dimensional)
                    X_prediction = Fhat_selected[-3:, :]
                    reshaped_X_prediction = X_prediction.T.flatten()

                    #For each regressor a polynomial
                    polys = ['expalmon'] * (X_prediction.shape[1])

                    # Estimating MIDAS-e model and making predictions in training sample
                    if Ylag is None:
                        result = estimate_general(Y, None, reshaped_X_historical[1:], polys, lambda_1)
                        Y_pred3_previous[ll] = forecast_general(reshaped_X_prediction, None, result, polys)
                        e_pred3_previous[ll] = actual_data_previous - Y_pred3_previous[ll]
                    else:
                        result = estimate_general(Y, Ylag, reshaped_X_historical[1:], polys, lambda_1)
                        Y_pred3_previous[ll] = forecast_general(reshaped_X_prediction, Ylag[-1].reshape(1, -1), result, polys)
                        e_pred3_previous[ll] = actual_data_previous - Y_pred3_previous[ll]

                    #-----------------

            # For the two predictions made in similar quarter the errors are extracted for each regression method
            e_pred1_selected = e_pred1_previous[[0, 3]]
            e_pred2_selected = e_pred2_previous[[0, 3]]
            e_pred3_selected = e_pred3_previous[[0, 3]]

            # The MSPE is stored for the each combination of hyperparameter and optimal parameters
            BigTable1[(hyper, kopt1, mopt1, popt1)] = np.mean((e_pred1_selected)**2)  # for BRIDGE
            BigTable2[(hyper, kopt2, mopt2, popt2)] = np.mean((e_pred2_selected)**2)  # for MIDAS1
            BigTable3[(hyper, kopt3, mopt3, popt3)] = np.mean((e_pred3_selected)**2)

            # The data kept in store is switched if it was the last hyperparameter in the grid
            if hypernum == len(grid) - 1:

                four_months_ago_monthly = three_months_ago_monthly
                four_months_ago_quarterly = three_months_ago_quarterly
                four_months_ego_quarters_to_go = three_months_ego_quarters_to_go
                four_months_ego_actual_data_previous = three_months_ego_actual_data_previous

                three_months_ago_monthly = two_months_ago_monthly
                three_months_ago_quarterly = two_months_ago_quarterly
                three_months_ego_quarters_to_go = two_months_ego_quarters_to_go
                three_months_ego_actual_data_previous = two_months_ego_actual_data_previous

                two_months_ago_monthly = one_months_ago_monthly
                two_months_ago_quarterly = one_months_ago_quarterly
                two_months_ego_quarters_to_go = one_months_ego_quarters_to_go
                two_months_ego_actual_data_previous = one_months_ego_actual_data_previous

                # For the month a new data set is extracted and imputed (as it is not ready made)
                prepare_current_data(current_month - 2, first_month_in_data, use_vintage_data)
                one_months_ago_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)
                one_months_ago_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)
                one_months_ego_quarters_to_go = last_vintage_data_quarterly.shape[0] - one_months_ago_quarterly.shape[0]
                one_months_ego_actual_data_previous =  last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - one_months_ego_quarters_to_go]

                # Again the unkown months in the quarter are imputed, similar as documented earlier
                added_rows = 0
                if t == 0:
                    added_rows = 2
                elif t == 1:
                    added_rows = 1
                else:
                    if (one_months_ago_monthly.shape[0] - referenceValue) % 3 == 0:
                        added_rows = 3
                    elif (one_months_ago_monthly.shape[0] - referenceValue) % 3 == 1:
                        added_rows = 2
                    else:
                        added_rows = 1
                for i in range(added_rows):
                    Nan_row = pd.DataFrame([[np.nan] * one_months_ago_monthly.shape[1]], columns=one_months_ago_monthly.columns)
                    one_months_ago_monthly = pd.concat([one_months_ago_monthly, Nan_row], ignore_index=True)

                    random_cols = random.sample(list(one_months_ago_monthly.columns), 13)
                    for col in random_cols:
                        one_months_ago_monthly.iloc[-1, one_months_ago_monthly.columns.get_loc(col)] = one_months_ago_monthly.iloc[-2, one_months_ago_monthly.columns.get_loc(col)]
                    one_months_ago_monthly.to_csv("/content/Monthly_Data_Modified.csv", header = None, index=False)
                    one_months_ago_monthly = Impute()
                    one_months_ago_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv", header=None)

            # ---------------------------------------------end

        # The model combination with the lowest SMPE is extracted
        hyperopt1, kopt1, mopt1, popt1 = min(BigTable1, key=BigTable1.get)
        hyperopt2, kopt2, mopt2, popt2 = min(BigTable2, key=BigTable2.get)
        hyperopt3, kopt3, mopt3, popt3 = min(BigTable3, key=BigTable3.get) #getting the best grid ones for the last 6 months


        # The model is now used to make the final prediction, similarly as documented earlier -------- start

        Fhat1 = getPCs(current_data_monthly, Mode, hyperopt1, K)
        Fhat2 = getPCs(current_data_monthly, Mode, hyperopt2, K)
        Fhat3 = getPCs(current_data_monthly, Mode, hyperopt3, K)


        # Final Bridge estimation and prediction --------------

        if popt1 == 0:
            Y = current_data_quarterly.values
            Y = Y[1:]
            Ylag = None
        else:
            [Y, Ylag] = lagaugment(current_data_quarterly.values, 1)
            Y = Y[1:]
            Ylag = Ylag[1:]

        Fhat_selected_original = Fhat1[:, :kopt1]

        nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan)
        Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

        [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mopt1)
        Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kopt1), Fhat_selectedLag[:, 0:-kopt1])) #combines the factors and the lags into a single set of factors

        Fhat_selected = pd.DataFrame(Fhat_selected)
        Fhat_selected.fillna(Fhat_selected.median(), inplace=True)
        Fhat_selected = Fhat_selected.to_numpy()

        number_of_quarters = Fhat_selected.shape[0] // 3
        reshaped_data = Fhat_selected.reshape(number_of_quarters, 3, Fhat_selected.shape[1])
        Fhat_averaged = np.nanmean(reshaped_data, axis=1)

        if Ylag is None:
            X_historical = Fhat_averaged[1:-1]
            X_prediction = np.hstack((np.array([[1]]), Fhat_averaged[-1].reshape(1, -1)))
        else:
            X_historical = np.hstack((Fhat_averaged[1:-1], Ylag))
            X_prediction = np.hstack((np.array([[1]]), Fhat_averaged[-1].reshape(1, -1), Ylag[-1].reshape(1, -1)))
        X_historical = sm.add_constant(X_historical)

        # Final estimation and prediction, and storring of the prediction, error and optimal parameters
        model = Ridge(alpha=lambda_1)
        model.fit(X_historical, Y)
        Y_pred1[t] = model.predict(X_prediction)
        e_pred1[t] = actual_data - Y_pred1[t]
        optimal_params1[t] = (kopt1, mopt1, popt1, hyperopt1)


        # Final MIDAS-e estimation and prediction --------------

        if popt2 == 0:
            Y = current_data_quarterly.values
            Y = Y[1:]
            Ylag = None
        else:
            [Y, Ylag] = lagaugment(current_data_quarterly.values, 1)
            Y = Y[1:]
            Ylag = Ylag[1:]

        Fhat_selected_original = Fhat2[:, :kopt2]

        nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan)
        Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

        [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mopt2)
        Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kopt2), Fhat_selectedLag[:, 0:-kopt2])) #combines the factors and the lags into a single set of factors

        Fhat_selected = pd.DataFrame(Fhat_selected)
        Fhat_selected.fillna(Fhat_selected.median(), inplace=True)
        Fhat_selected = Fhat_selected.to_numpy()

        X_historical = Fhat_selected[:-3, :]
        X_prediction = np.hstack( Fhat_selected[-3:].reshape(3, Fhat_selected.shape[1]))

        number_of_regressors = X_historical.shape[1]
        months = 3
        reshaped_X_historical = np.zeros((X_historical.shape[0] // months, number_of_regressors*months))
        for j in range(number_of_regressors):
            current_start = 0
        for i in range(X_historical.shape[0]):
            current_row = i // months
            current_addition = i % months
            reshaped_X_historical[current_row, j*months + current_addition] = X_historical[i, j]

        X_prediction = Fhat_selected[-3:, :]
        reshaped_X_prediction = X_prediction.T.flatten()

        polys = ['beta'] * (X_prediction.shape[1])

         # Final estimation and prediction, and storring of the prediction, error and optimal parameters
        if Ylag is None:
            result = estimate_general(Y, None, reshaped_X_historical[1:], polys, lambda_1)
            Y_pred2[t] = forecast_general(reshaped_X_prediction, None, result, polys)
            e_pred2[t] = actual_data - Y_pred2[t]
        else:
            result = estimate_general(Y, Ylag, reshaped_X_historical[1:], polys, lambda_1)
            Y_pred2[t] = forecast_general(reshaped_X_prediction, Ylag[-1].reshape(1, -1), result, polys)
            e_pred2[t] = actual_data - Y_pred2[t]
        optimal_params2[t] = (kopt2, mopt2, popt2, hyperopt2)




        # Final MIDAS-e estimation and prediction --------------

        if popt3 == 0:
            Y = current_data_quarterly.values
            Y = Y[1:]
            Ylag = None
        else:
            [Y, Ylag] = lagaugment(current_data_quarterly.values, 1)
            Y = Y[1:]
            Ylag = Ylag[1:]

        Fhat_selected_original = Fhat3[:, :kopt3]

        nan_row = np.full((1, Fhat_selected_original.shape[1]), np.nan)
        Fhat_selected = np.vstack((Fhat_selected_original, nan_row))

        [Fhat_selected, Fhat_selectedLag] = lagaugment(Fhat_selected[:-1], mopt3)
        Fhat_selected = np.hstack((Fhat_selected.reshape(Fhat_selected.shape[0], kopt3), Fhat_selectedLag[:, 0:-kopt3]))

        Fhat_selected = pd.DataFrame(Fhat_selected)
        Fhat_selected.fillna(Fhat_selected.median(), inplace=True)
        Fhat_selected = Fhat_selected.to_numpy()

        X_historical = Fhat_selected[:-3, :]
        X_prediction = np.hstack( Fhat_selected[-3:].reshape(3, Fhat_selected.shape[1]))

        number_of_regressors = X_historical.shape[1]
        months = 3
        reshaped_X_historical = np.zeros((X_historical.shape[0] // months, number_of_regressors*months))
        for j in range(number_of_regressors):
            current_start = 0
        for i in range(X_historical.shape[0]):
            current_row = i // months
            current_addition = i % months
            reshaped_X_historical[current_row, j*months + current_addition] = X_historical[i, j]

        X_prediction = Fhat_selected[-3:, :]
        reshaped_X_prediction = X_prediction.T.flatten()

        polys = ['expalmon'] * (X_prediction.shape[1])

        # Final estimation and prediction, and storring of the prediction, error and optimal parameters
        if Ylag is None:
            result = estimate_general(Y, None, reshaped_X_historical[1:], polys, lambda_1)
            Y_pred3[t] = forecast_general(reshaped_X_prediction, None, result, polys)
            e_pred3[t] = actual_data - Y_pred3[t]
        else:
            result = estimate_general(Y, Ylag, reshaped_X_historical[1:], polys, lambda_1)
            Y_pred3[t] = forecast_general(reshaped_X_prediction, Ylag[-1].reshape(1, -1), result, polys)
            e_pred3[t] = actual_data - Y_pred3[t]

        optimal_params3[t] = (kopt3, mopt3, popt3, hyperopt3)

        # ---------------------------------------------end

        # Putting parameters in arrays for storing
        optimal_params1 = np.atleast_1d(optimal_params1)
        optimal_params2 = np.atleast_1d(optimal_params2)
        optimal_params3 = np.atleast_1d(optimal_params3)

        # Filtering parameters for NaN as it leads to errors when saving
        filtered_params1 = [t for t in optimal_params1 if t is not None]
        filtered_params2 = [t for t in optimal_params2 if t is not None]
        filtered_params3 = [t for t in optimal_params3 if t is not None]

        # Putting predictions in arrays for storing
        Y_pred1 = np.atleast_1d(Y_pred1)
        Y_pred2 = np.atleast_1d(Y_pred2)
        Y_pred3 = np.atleast_1d(Y_pred3)
        e_pred1 = np.atleast_1d(e_pred1)
        e_pred2 = np.atleast_1d(e_pred2)
        e_pred3 = np.atleast_1d(e_pred3)

        # Putting parameters in list for storing
        optimal_params_2d1 = np.array([list(t) for t in filtered_params1])
        optimal_params_2d2 = np.array([list(t) for t in filtered_params2])
        optimal_params_2d3 = np.array([list(t) for t in filtered_params3])

        # Saving all results within each iteration for safety precautions
        if Mode[0] == 'P' or Mode[0] == 'S':
            np.savetxt('/content/' + Mode + '_pred_Bridge.csv', Y_pred1, delimiter=',')
            np.savetxt('/content/' + Mode + '_pred_MIDAS-b.csv', Y_pred2, delimiter=',')
            np.savetxt('/content/' + Mode + '_pred_MIDAS-e.csv', Y_pred3, delimiter=',')
            np.savetxt('/content/' + Mode + '_error_Bridge.csv', e_pred1, delimiter=',')
            np.savetxt('/content/' + Mode + '_error_MIDAS-b.csv', e_pred2, delimiter=',')
            np.savetxt('/content/' + Mode + '_error_MIDAS-e.csv', e_pred3, delimiter=',')
            np.savetxt('/content/' + Mode + '_optimal_ParamsBridge.csv', optimal_params_2d1, delimiter=',', fmt='%.18e')
            np.savetxt('/content/' + Mode + '_optimal_ParamsMIDAS-b.csv', optimal_params_2d2, delimiter=',', fmt='%.18e')
            np.savetxt('/content/' + Mode + '_optimal_ParamsMIDAS-e.csv', optimal_params_2d3, delimiter=',', fmt='%.18e')
        elif Mode[0] == 'K':
            np.savetxt('/content/' + Mode + '_pred_Bridge.csv', Y_pred1, delimiter=',')
            np.savetxt('/content/' + Mode + '_pred_MIDAS-b.csv', Y_pred2, delimiter=',')
            np.savetxt('/content/' + Mode + '_pred_errorMIDAS-e.csv', Y_pred3, delimiter=',')
            np.savetxt('/content/' + Mode + '_error_Bridge.csv', e_pred1, delimiter=',')
            np.savetxt('/content/' + Mode + '_error_MIDAS-b.csv', e_pred2, delimiter=',')
            np.savetxt('/content/' + Mode + '_error_MIDAS-e.csv', e_pred3, delimiter=',')
            np.savetxt('/content/' + Mode + '_optimal_ParamsBridge.csv', optimal_params_2d1, delimiter=',', fmt='%.18e')
            np.savetxt('/content/' + Mode + '_optimal_ParamsMIDAS-b.csv', optimal_params_2d2, delimiter=',', fmt='%.18e')
            np.savetxt('/content/' + Mode + '_optimal_ParamsMIDAS-e.csv', optimal_params_2d3, delimiter=',', fmt='%.18e')

    # Final MSPE is calculated
    MSPE1 = np.mean((e_pred1)**2)
    MSPE2 = np.mean((e_pred2)**2)
    MSPE3 = np.mean((e_pred3)**2)

    # Put everything in arrays for storing
    MSPE1 = np.atleast_1d(MSPE1)
    MSPE2 = np.atleast_1d(MSPE2)
    MSPE3 = np.atleast_1d(MSPE3)
    Y_pred1 = np.atleast_1d(Y_pred1)
    Y_pred2 = np.atleast_1d(Y_pred2)
    Y_pred3 = np.atleast_1d(Y_pred3)
    e_pred1 = np.atleast_1d(e_pred1)
    e_pred2 = np.atleast_1d(e_pred2)
    e_pred3 = np.atleast_1d(e_pred3)
    optimal_params1 = np.atleast_1d(optimal_params1)
    optimal_params2 = np.atleast_1d(optimal_params2)
    optimal_params3 = np.atleast_1d(optimal_params3)

    # Putting parameters in list for storing
    optimal_params_2d1 = np.array([list(t) for t in optimal_params1])
    optimal_params_2d2 = np.array([list(t) for t in optimal_params2])
    optimal_params_2d3 = np.array([list(t) for t in optimal_params3])

    # Final safe of all results
    if Mode[0] == 'P' or Mode[0] == 'S':
        np.savetxt('/content/' + Mode + '_MSPEs_Bridge.csv', MSPE1, delimiter=',')
        np.savetxt('/content/' + Mode + '_MSPEs_MIDAS-b.csv', MSPE2, delimiter=',')
        np.savetxt('/content/' + Mode + '_MSPEs_MIDAS-e.csv', MSPE3, delimiter=',')
        np.savetxt('/content/' + Mode + '_pred_Bridge.csv', Y_pred1, delimiter=',')
        np.savetxt('/content/' + Mode + '_pred_MIDAS-b.csv', Y_pred2, delimiter=',')
        np.savetxt('/content/' + Mode + '_pred_MIDAS-e.csv', Y_pred3, delimiter=',')
        np.savetxt('/content/' + Mode + '_error_Bridge.csv', e_pred1, delimiter=',')
        np.savetxt('/content/' + Mode + '_error_MIDAS-b.csv', e_pred2, delimiter=',')
        np.savetxt('/content/' + Mode + '_error_MIDAS-e.csv', e_pred3, delimiter=',')
        np.savetxt('/content/' + Mode + '_optimalParams_Bridge.csv', optimal_params_2d1, delimiter=',', fmt='%.18e')
        np.savetxt('/content/' + Mode + '_optimalParams_MIDAS-b.csv', optimal_params_2d2, delimiter=',', fmt='%.18e')
        np.savetxt('/content/' + Mode + '_optimalParams_MIDAS-e.csv', optimal_params_2d3, delimiter=',', fmt='%.18e')
    elif Mode[0] == 'K':
        np.savetxt('/content/' + Mode + '_MSPEs_Bridge.csv', MSPE1, delimiter=',')
        np.savetxt('/content/' + Mode + '_MSPEs_MIDAS-b.csv', MSPE2, delimiter=',')
        np.savetxt('/content/' + Mode + '_MSPEs_MIDAS-e.csv', MSPE3, delimiter=',')
        np.savetxt('/content/' + Mode + '_pred_Bridge.csv', Y_pred1, delimiter=',')
        np.savetxt('/content/' + Mode + '_pred_MIDAS-b.csv', Y_pred2, delimiter=',')
        np.savetxt('/content/' + Mode + '_pred_MIDAS-e.csv', Y_pred3, delimiter=',')
        np.savetxt('/content/' + Mode + '_error_Bridge.csv', e_pred1, delimiter=',')
        np.savetxt('/content/' + Mode + '_error_MIDAS-b.csv', e_pred2, delimiter=',')
        np.savetxt('/content/' + Mode + '_error_MIDAS-e.csv', e_pred3, delimiter=',')
        np.savetxt('/content/' + Mode + '_optimalParams_Bridge.csv', optimal_params_2d1, delimiter=',', fmt='%.18e')
        np.savetxt('/content/' + Mode + '_optimalParams_MIDAS-b.csv', optimal_params_2d2, delimiter=',', fmt='%.18e')
        np.savetxt('/content/' + Mode + '_optimalParams_MIDAS-e.csv', optimal_params_2d3, delimiter=',', fmt='%.18e')
