'''This module contains all functions needed for the main program to run.'''

#Several imports from predefined Python functions.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import tempfile
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
import time
from numpy.random import seed
import random
from rpy2.robjects import r, pandas2ri, FloatVector, DataFrame
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from scipy.optimize import least_squares
from midas.weights import polynomial_weights, WeightMethod, BetaWeights, ExpAlmonWeights #midaspy
from midas.fit import jacobian_wx #midaspy

pandas2ri.activate() #activates R

def lagaugment(a, p):  # Function by Kutadelutze (2022), full reference can be found in main article

    '''
    Extends the original matrix with its lags.

            Parameters:
                    a (NumPy array): The original matrix
                    p (int): The number of lags the matrix is expanded by

            Returns:
                    a, b (NumPy array): Matrix consisting of the original matrix and its lags
    '''

    try:
        a = a.reshape(a.shape[0], 1)
    except ValueError or IndexError:
        pass
    if p == 0:
        return a
    else:
        b = np.zeros([a.shape[0], p * a.shape[1]])
        b.fill(np.NaN)
        for pp in range(p):
            b[pp + 1:, a.shape[1] * pp:(pp + 1) * a.shape[1]] = a[:-(pp + 1), :]
    return a, b


def dmtest(e1, e2, h):  # Function by Kutadelutze (2022), full reference can be found in main article

    '''
    Calculates the Diebold Mariano (DM) test.

            Parameters:
                    e1 (NumPy array): The original matrix
                    e2 (NumPy array): The number of lags the matrix is expanded by
                    h (int): The number of horizons, equals 1 when nowcasting GDP

            Returns:
                    DM (double): The Diebold-Mariano test statistic
    '''

    e1 = e1.reshape(e1.shape[0], 1)
    e2 = e2.reshape(e2.shape[0], 1)
    T = e1.shape[0]  # Observations
    d = e1**2 - e2**2  # Loss differential
    Mu = np.mean(d)  # Var of Loss differential
    gamma0 = np.var(d) * T / (T - 1)  # Autocorrelation

    if h > 1:
        gamma = np.zeros([h - 1, 1])
        for i in range(1, h):
            sCov = np.cov(np.vstack((d[i:T].T, d[0:T - i].T)))
            gamma[i - 1] = sCov[0, 1]
        varD = gamma0 + 2 * sum(gamma)[0]
    else:
        varD = gamma0

    DM = Mu / np.sqrt((1 / T) * varD)  # DM statistic ~N(0,1)
    return DM

def getPCs(data_training, Mode, hyper, K):

    '''
    Returns the principal componets associated with the data.

            Parameters:
                    data_training (DataFrame): The monthly regressor data
                    Mode (List): Contains the type of principal component to be returned
                    hyper (double): Is the hyperparamater value for the principal component
                    K (int): The number of PCs to be returned

            Returns:
                    Fhat (DataFrame): Matrix consisting of the principal components
    '''

    # Standardize the training data
    data_training_standardized = stats.zscore(data_training, axis=0, ddof=1)

    # Determine the kernel type if applicable
    if Mode[0] == 'K':
        kernel = Mode[3:]  # Extracts kernel type
    else:
        kernel = False

    # Initialize the transformed data variable
    Fhat = None

    # PCA
    if Mode[0] == 'P':
      pca = PCA(n_components=K)
      Fhat = pca.fit_transform(data_training_standardized)  # Calculates the linear principal components

    # SPC
    elif Mode[0] == 'S':
      Wts = np.hstack((data_training_standardized, data_training_standardized**2))
      Wts = StandardScaler().fit_transform(Wts)
      pca = PCA(n_components=K)
      Fhat = pca.fit_transform(Wts)

    # KPCA
    elif Mode[0] == 'K':
        if kernel in ['sigmoid', 'rbf']:
          transformer = KernelPCA(n_components=K, kernel=kernel, gamma=hyper)
        else:
            transformer = KernelPCA(n_components=K, kernel=kernel, degree=2)
        Fhat = transformer.fit_transform(data_training_standardized)

    return Fhat


def prepare_current_data(m, first_month_in_data, use_vintage_data):

    '''
    Extracts the current data and fully processes it.

            Parameters:
                    m (int): The current vintage data requested
                    first_month_in_data (int): The first month in the training data
                    use_vintage_data (bool): Whether or not vintage data is used

            Returns:
                    None: The processed data is not returned but for safety explicitly written to a csv file
    '''

    q = (m - 1) // 3 # The current quarter
    year = 1960 + (m - 1) // 12 # The current year
    month = (m - 1) % 12 + 1 # The current vintage month (differs from m)
    monthly_str = f"{year}-{month:02d}m.csv" # String is formed to extract the correct monthly vintage data later

    # Opens all files for data processing and makes them empty. Modified data is for intermediate processing and transformed data is final processed data.
    with open('/content/Monthly_Data_Modified.csv', 'w') as file:
        pass
    with open('/content/Quarterly_Data_Modified.csv', 'w') as file:
        pass
    with open('/content/Transformed_Monthly_Data.csv', 'w') as file:
        pass
    with open('/content/Transformed_Quarterly_Data.csv', 'w') as file:
        pass

    if use_vintage_data:
        print(monthly_str)
        monthly_file_path = f"/content/{monthly_str}"
        quarterly_file_path = f"/content/Quarterly_Vintages.xlsx"
        Monthly = pd.read_csv(monthly_file_path, header=None) # the current monthly data is read
        Quarterly = pd.read_excel(quarterly_file_path, header=None) # the quarterly data is read
        new_monthly_file_path = "/content/Monthly_Data_Modified.csv"
        new_quarterly_file_path = "/content/Quarterly_Data_Modified.csv"

        # The current quarterly vintage data is extracted by selecting the correct column
        colomQuarterly = m - 69  # Small correction of -69 is done to match the month with the columns of the dataset
        Quarterly = Quarterly.iloc[:, colomQuarterly]

        # The quarterly data is cut to exclude periods after the current month
        end_index = Quarterly.last_valid_index()
        Quarterly = Quarterly.iloc[:end_index + 1]

        # Saving adjusted dataframe to a new Excel (to not alter orginal dataset)
        Quarterly = Quarterly.to_frame()
        Monthly.to_csv("/content/Monthly_Data_Modified.csv", header=None, index=False)
        Quarterly.to_csv("/content/Quarterly_Data_Modified.csv", header=None, index=False)

        # Transforming data
        transformation_function("/content/Monthly_Data_Modified.csv", "/content/Quarterly_Data_Modified.csv", first_month_in_data, m)

        # Makes the data ready to be used by EM algorithm
        Setup_Imputation()

        # EM algorithm
        Impute()

    else:
        monthly_file_path = "/content/2024-04m.csv" # Last monthly vintage data is always selected
        quarterly_file_path = f"/content/Quarterly_Vintages.xlsx"
        Monthly = pd.read_csv(monthly_file_path, header=None)
        Quarterly = pd.read_excel(quarterly_file_path, header=None)
        new_monthly_file_path = "/content/Monthly_Data_Modified.csv"
        new_quarterly_file_path = "/content/Quarterly_Data_Modified.csv"

        #The vintage data is accessed to check where to cut off the last vintage data sets
        monthly_file_path = f"/content/{monthly_str}"
        quarterly_file_path = f"/content/Quarterly_Vintages.xlsx"
        Monthly_reference = pd.read_csv(monthly_file_path, header=None)
        Quarterly_reference = pd.read_excel(quarterly_file_path, header=None)
        print(monthly_str)

        # The current quarterly vintage data is extracted by selecting the correct column
        colomQuarterly = m - 69  # Small correction of -69 is done to match the month with the columns of the dataset
        Quarterly = Quarterly.iloc[:, colomQuarterly]

        # The quarterly data is cut to exclude periods after the current month
        end_index = Quarterly.last_valid_index()
        Quarterly = Quarterly.iloc[:end_index + 1]

        # Cuts the last vintage data to match it with the real vintage data size
        monthly_diff = len(Monthly) - len(Monthly_reference)
        quarterly_diff = len(Quarterly) - len(Quarterly_reference)
        if monthly_diff > 0:
            Monthly = Monthly.iloc[:-monthly_diff]
        if quarterly_diff > 0:
            Quarterly = Quarterly.iloc[:-quarterly_diff]

        # Initialising categories for variables who have lags in the real vintage data (grouped by colomn number)
        cols_zero_or_one = [116, 117, 118, 119]
        cols_one = [58, 62, 63, 72]
        cols_one_or_two = [1, 2, 3, 75, 124, 125]
        cols_zero_to_eight = [20, 21, 76]

        # Setting last zero or one elements to zero for columns 116, 117, 118, and 119
        for col in cols_zero_or_one:
            num_to_zero = np.random.randint(0, 2) # Random number generator to determine whether put to zero or one
            Monthly.iloc[-num_to_zero:, col] = 0

        # Setting last element to zero for columns 58, 62, 63, and 72
        for col in cols_one:
            Monthly.iloc[-1, col] = 0

        # Setting last one or two elements to zero for columns 1, 2, 3, 75, 124, 125
        for col in cols_one_or_two:
            num_to_zero = np.random.randint(1, 3)
            Monthly.iloc[-num_to_zero:, col] = 0

        # Setting last zero up to eight elements to zero for columns 20, 21, and 76
        for col in cols_zero_to_eight:
            num_to_zero = np.random.randint(0, 9)
            Monthly.iloc[-num_to_zero:, col] = 0


        # Saving adjusted dataframe to a new Excel (to not alter orginal dataset)
        Quarterly = Quarterly.to_frame()
        Monthly.to_csv("/content/Monthly_Data_Modified.csv", header=None, index=False)
        Quarterly.to_csv("/content/Quarterly_Data_Modified.csv", header=None, index=False)

        # Transforming data
        transformation_function("/content/Monthly_Data_Modified.csv", "/content/Quarterly_Data_Modified.csv", first_month_in_data, m)


        # Makes the data ready to be used by EM algorithm
        Setup_Imputation()

        # EM algorithm
        Impute()

        return None


def transformation_function(new_monthly_file_path, new_quarterly_file_path, first_month_in_data, m): # Function partly from Benett (2023), full reference can be found in main article

    '''
    Transforms the data.

            Parameters:
                    new_monthly_file_path (string): The file of the monthly data
                    new_quarterly_file_path (string): The file of the quarterly data
                    first_month_in_data (int): The first month in the training data
                    m (int): the current month

            Returns:
                    None: The processed data is not returned but for safety explicitly written to a csv file
    '''

    # Data loaded
    Monthly = pd.read_csv(new_monthly_file_path, header=None)
    Quarterly = pd.read_csv(new_quarterly_file_path, header=None)

    # The last starting and current quarter are calculated
    first_quarter_in_data = (first_month_in_data - 1) // 3
    q = (m - 1) // 3 + 1 #added one later to make it correct, might have to check for different months in debugging

    # Cutting monthly data
    rawdata1 = Monthly.values[13+first_month_in_data -1:13+m, 1:]  # extract raw data used for transformations later
    Trans_code1 = Monthly.values[1, 1:].astype(int).reshape(-1, 1)  # extract transformation codes

    # Cutting quarterly data
    rawdata2 = Quarterly.values[53 + first_quarter_in_data - 1:, 0]   # making sure we start at the right moment

    # Initialising the transformed data matrices
    X = np.full((rawdata1.shape[0], rawdata1.shape[1]), np.nan)
    small = 1e-6  # Threshold for small values

    # Loop that for all variables transforms the data
    for i in range(rawdata1.shape[1]):
        x = rawdata1[:, i].astype(float)
        tcode = int(Trans_code1[i][0])  # Ensure tcode is integer
        n = x.shape[0]

        if tcode == 1:  # Level (i.e. no transformation): x(t)
            X[:, i] = x
        elif tcode == 2:  # First difference: x(t)-x(t-1)
            X[1:n, i] = x[1:n] - x[0:n-1]
        elif tcode == 3:  # Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
            X[2:n, i] = x[2:n] - 2 * x[1:n-1] + x[0:n-2]
        elif tcode == 4:  # Natural log: ln(x)
            for j in range(n):
                if x[j] > small:
                    X[j, i] = np.log(x[j])
                else:
                    X[j, i] = np.nan
        elif tcode == 5:  # First difference of natural log: ln(x)-ln(x-1)
            for j in range(1, n):
                if x[j] > small and x[j-1] > small:
                    X[j, i] = np.log(x[j]) - np.log(x[j-1])
                else:
                    X[j, i] = np.nan
        elif tcode == 6:  # Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
            for j in range(2, n):
                if x[j] > small and x[j-1] > small and x[j-2] > small:
                    X[j, i] = np.log(x[j]) - 2 * np.log(x[j-1]) + np.log(x[j-2])
                else:
                    X[j, i] = np.nan
        elif tcode == 7:  # First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
            z = np.zeros([n])
            for j in range(1, n):
                if x[j-1] > small:
                    z[j] = (x[j] - x[j-1]) / x[j-1]
            for j in range(2, n):
                X[j, i] = z[j] - z[j-1]

    # Initialising the transformed data matrices
    Y = np.full((rawdata2.shape[0], 1), np.nan)
    small = 1e-6  # small value to avoid log of zero

    # Transform the GDP via growth rate log (transformation code 5)
    x = rawdata2.astype(float)
    n = x.shape[0]
    for j in range(1, n):
        if x[j] > small and x[j-1] > small:
            Y[j, 0] = np.log(x[j]) - np.log(x[j-1])
        else:
            Y[j, 0] = np.nan

    # Save transformed data
    np.savetxt("/content/Monthly_Data_Modified.csv", X, delimiter=',')
    np.savetxt("/content/Quarterly_Data_Modified.csv", Y, delimiter=',')

    return None

def Setup_Imputation():

    '''
    Sets up the data such that it can be imputed after.

            Parameters:
                    None: The function only uses the data from the Excel document

            Returns:
                    None: The processed data is not returned but for safety explicitly written to a csv file
    '''

    # Data is read
    Monthly = pd.read_csv("/content/Monthly_Data_Modified.csv", header=None)
    Quarterly = pd.read_csv("/content/Quarterly_Data_Modified.csv", header=None)

    # The columns that only NaN are deleted as it interferes with EM algorithm
    Monthly = Monthly.dropna(axis=1, how='all')

    # First row for the monthly and quarterly data is dropped as it contains only NaN values after the earlier transformation
    Monthly = Monthly.drop(index=0)
    Quarterly = Quarterly.drop(index=0)

    # All zero values are replaced by NaN values as is needed for EM algorithm and is needed in general
    Monthly.replace(0, np.nan, inplace=True)
    Quarterly.replace(0, np.nan, inplace=True)

    # We convert Quarterly DataFrame to a Series for the the AR imputation.
    Quarterly = Quarterly.iloc[:, 0]

    # Quarterly Data is AR(1) imputed by the last known element
    if Quarterly.isnull().any():
        Quarterly.fillna(method='ffill', inplace=True)

    # We convert it back to a DataFrame to save it
    Quarterly = Quarterly.to_frame()

    #Saving data
    Monthly.to_csv("/content/Monthly_Data_Modified.csv", header = None, index=False)
    Quarterly.to_csv("/content/Transformed_Quarterly_Data.csv", header = None, index=False)

    return None


def Impute():

    '''
    Imputes the data.

            Parameters:
                    None: The function only uses the data from the Excel document

            Returns:
                    None: The processed data is not returned but for safety explicitly written to a csv file
    '''

    Monthly = pd.read_csv("/content/Monthly_Data_Modified.csv", header=None)

    # Putting the data in a format that R can work with later
    Monthly_R_format = DataFrame({col: FloatVector(Monthly[col].astype(float).values) for col in Monthly.columns})

    # Assigning the data to R
    r.assign('Monthly', Monthly_R_format)

    # R code
    r_code = """
    # Installs several packages which are needed for EM algorithm, only if they are not yet installed
    if (!requireNamespace("devtools", quietly = TRUE)) {
        install.packages("devtools", repos='http://cran.us.r-project.org')
    }
    if (!requireNamespace("pracma", quietly = TRUE)) {
        install.packages("pracma", repos='http://cran.us.r-project.org')
    }
    if (!requireNamespace("readr", quietly = TRUE)) {
        install.packages("readr", repos='http://cran.us.r-project.org')
    }
    if (!requireNamespace("stats", quietly = TRUE)) {
        install.packages("stats", repos='http://cran.us.r-project.org')
    }
    library(devtools)
    #library(pracma)
    library(readr)
    library(stats)

    # Installs EM algorithm if not yet installed
    if (!requireNamespace("fbi", quietly = TRUE)) {
        devtools::install_github("cykbennie/fbi")
    }
    library(fbi)

    # Function that transforms the data
    transform_dataset <- function(Monthly) {

        kmax <- 12
        if (!is.matrix(Monthly)) {
        Monthly <- as.matrix(Monthly)
        }

        # EM algorithm as per Benett (2023)
        transformed_monthly <- tp_apc(Monthly, kmax = kmax, center = TRUE, standardize = TRUE, re_estimate = TRUE)

        # Save results to the excel sheet
        write.csv(as.data.frame(transformed_monthly$data), '/content/Transformed_Monthly_Data.csv', row.names = FALSE)
    }

    transform_dataset(Monthly)

    """

    # Executes the R code and saves it a second time which is needed as in rare cases the changes to the excel sheet are not kept
    r(r_code)
    transformed_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv")
    np.savetxt("/content/Transformed_Monthly_Data.csv", transformed_monthly, delimiter=',')

    return None


def estimate_general(y, yl, X, polys, lambda_1): # Function is an extended version of the functions from Zuskin (2020), and Sapphire (2020), see article for full references

    '''
    Estimates a MIDAS model using NLS.

            Parameters:
                    y: The dependent variable (growth rate of GDP)
                    yl: The first lag of the dependent variable (growth rate of GDP)
                    X: The regressor data set
                    polys: The polynomial weighting methods that are used
                    lambda_1: Ridge penalty on regressors of X to prevent overfitting of MIDAS

            Returns:
                    opt_res: The estimated parameters
    '''
    # Extracts the weighting methods
    weight_methods = [polynomial_weights(poly) for poly in polys]

    # Based on the number of weight_methods the number of regressors in X is determined
    num_regressors = len(weight_methods)

    # The data is transformed according to the weighting method and its initialsed weights
    xws = [weight_method.x_weighted(X[:, i*3:(i+1)*3], weight_method.init_params())[0] for i, weight_method in enumerate(weight_methods)]

    # Channging the data from a list of arrays to a 2D numpy array, which is needed for the next step
    xw_concat = np.concatenate([xw.reshape((len(xw), 1)) for xw in xws], axis=1)

    if yl is not None: # one lag of y

        # First we do OLS to get initial parameters
        c = np.linalg.lstsq(np.concatenate([np.ones((len(xw_concat), 1)), xw_concat, yl], axis=1), y, rcond=None)[0]

        # Initialisation of the objective function and Jacobian function for NLS later
        f = lambda v: ssr_generalized(v, X, y, yl, weight_methods, lambda_1)
        jac = lambda v: jacobian_generalized(v, X, y, yl, weight_methods)

        # Flattening c as it needs to a single dimension array
        c_t = c.T
        c_flat = c_t.flatten()

        # Concatenates the initial OLS estimates of X to the polynomial weights and to OLS estimates of ylag
        init_params = np.concatenate([c_flat[0:num_regressors + 1]] + [weight_method.init_params() for weight_method in weight_methods] + [c_flat[num_regressors + 1:]])

    else: # no lag of y

        # First we do OLS to get initial parameters
        c = np.linalg.lstsq(np.concatenate([np.ones((len(xw_concat), 1)), xw_concat], axis=1), y, rcond=None)[0]

        # Initialisation of the objective function and Jacobian function for NLS later
        f = lambda v: ssr_generalized(v, X, y, yl, weight_methods, lambda_1)
        jac = lambda v: jacobian_generalized(v, X, y, yl, weight_methods)

        # Flattening c as it needs to a single dimension array
        c_t = c.T
        c_flat = c_t.flatten()

        # Concatenates the initial OLS estimates of X to the polynomial weights
        init_params = np.concatenate([c_flat[0:num_regressors + 1]] + [weight_method.init_params() for weight_method in weight_methods])

    if isinstance(weight_methods[0], BetaWeights): # Beta polynomial requires bounds to ensure theta1 and theta2 > 0 (not the case for kPCA for high hyperparamter values)
        # Bounds for the various paramaters initialised
        lower_bounds = []
        upper_bounds = []

        # Infinite bounds for intercept and OLS coefficient regressors
        lower_bounds.extend([-np.inf] * (1 + num_regressors))
        upper_bounds.extend([np.inf] * (1 + num_regressors))

        # Bounds for theta1 and theta2
        for i in range(num_regressors):
            lower_bounds.extend([1e-6, 1e-6])  # Bounds for theta1 and theta2
            upper_bounds.extend([np.inf, np.inf])

        # Bounds for coefficient yl if it is not None
        if yl is not None:
            lower_bounds.extend([-np.inf] * len(yl[0]))
            upper_bounds.extend([np.inf] * len(yl[0]))

        # NLS optimisation for Beta polynomial
        opt_res = least_squares(f, init_params, jac, bounds=(lower_bounds, upper_bounds), xtol=1e-9, ftol=1e-9, max_nfev=5000, verbose=0)
    else:

        # NLS optimisation for Beta polynomial
        opt_res = least_squares(f, init_params, jac, xtol=1e-9, ftol=1e-9, max_nfev=5000, verbose=0)

    return opt_res


def forecast_general(Xfc, yfcl, res, polys): # Function is an extended version of the functions from Zuskin (2020), and Sapphire (2020), see article for full references

    '''
    Makes a forecast using the MIDAS model.

            Parameters:
                    Xfc: The factor values for the current quarter
                    yfcl: Previous value growth of GDP
                    res: The estimated parameters
                    polys: The polynomial weighting methods that are used

            Returns:
                    yf: The prediction made
    '''

    # Extracts the weighting methods
    weight_methods = [polynomial_weights(poly) for poly in polys]

    # Extracts the optimal MIDAS coefficients estimated earlier
    a = res.x[0]
    num_regressors = len(weight_methods)
    bs = res.x[1:num_regressors + 1]
    thetas = res.x[num_regressors + 1:num_regressors + 1 + 2 * num_regressors]
    lambdas = res.x[num_regressors + 1 + 2 * num_regressors:]


    #making the forecast(s) using the intercept, the factors and possibly lag of y
    if Xfc.ndim == 1: # In case only a single forecast is made
        yf = a
        for i, weight_method in enumerate(weight_methods):
            theta1, theta2 = thetas[2 * i:2 * i + 2]
            xw, _ = weight_method.x_weighted(Xfc[i * 3:(i + 1) * 3], [theta1, theta2])
            yf += bs[i] * xw

        if yfcl is not None:
            for i in range(len(lambdas)):
                yf += lambdas[i] * yfcl[i]

        return yf

    elif Xfc.ndim == 2: # In case of multiple forecasts (like for training data)
        nof = Xfc.shape[0]  # Number of forecasts
        yfs = np.zeros((nof, 1))

        for j in range(nof):
            yf = a
            for i, weight_method in enumerate(weight_methods):
                theta1, theta2 = thetas[2 * i:2 * i + 2]
                xw, _ = weight_method.x_weighted(Xfc[j, i * 3:(i + 1) * 3], [theta1, theta2])
                yf += bs[i] * xw

            if yfcl is not None:
                for i in range(len(lambdas)):
                    yf += lambdas[i] * yfcl[j, i]
            yfs[j, 0] = yf

        return yfs


def new_x_weighted(self, x, params): # Function is an extended version of the functions from Zuskin (2020), and Sapphire (2020), see article for full references

    '''
    Transforms the higher-frequency data into lower-frequency.

            Parameters:
                    self: The Beta polynomial for the higher-frequency regressor
                    x: The higher-frequency regressor
                    params: The current parameters

            Returns:
                    result: The lower-frequency data
    '''
    self.theta1, self.theta2 = params

    # Ensure x is 2D if it is 1D and make sure it's horizontal
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Weights are extracted via polynomial coefficients (x.shape[1] = M, the number of months per quarter)
    w = self.weights(x.shape[1])

    # Data is transformed
    result = np.dot(x, w)

    # Convert single-element array to a scalar
    if result.size == 1:
        result = result.item()

    return result, np.tile(w.T, (x.shape[0], 1))



def ssr_generalized(a, X, y, yl, weight_methods, lambda_1):

    '''
    Transforms the higher-frequency data into lower-frequency.

            Parameters:
                    a: The current OLS estimates of X, the current polynomial weights, and possibly the current ylag OLS estimate
                    X: The high-frequency data set
                    y: The dependent variable (growth rate of GDP)
                    yl: The first lag of the dependent variable (growth rate of GDP)
                    weight_methods: The polynomial weighting methods that are used
                    lambda_1: Ridge penalty on regressors of X to prevent overfitting of MIDAS

            Returns:
                    objective_value: The objective value outcome
    '''

    # Number of regressors is equal to number of weighting methods
    num_regressors = len(weight_methods)

    # Initialises a list for storing products between regressors and their estimates
    products = []

    # Intialises a list of the theta_parameter values
    theta_params = []

    # For all regressors the lower-frequency variables of X are multiplied by the estimate
    for i in range(num_regressors):
        theta_start = 1 + i * 2 + num_regressors
        theta_end = theta_start + 2
        theta_params.extend(a[theta_start:theta_end])

        xw, _ = weight_methods[i].x_weighted(X[:, i*3:(i+1)*3], a[theta_start:theta_end])
        products.append(a[1 + i] * xw)

    # The error made is the actual minus the prediction
    error = y - a[0] - sum(products)

    # Adding the product of ylag with its coefficient if it exists
    if yl is not None:
        error -= a[num_regressors]*yl

    SSR = sum(error**2)

    # Initialisation L2 penalty term on theta1 and theta2 which is needed for convergence
    convergence_paramater = 0.01   # Small penalty on the theta parameters needed for convergence

    # Objective value is SSR (OLS), the penalty on theta1 and theta2, and the ridge penalty
    objective_value = SSR + lambda_1 * np.sum(np.array(a[1:num_regressors + 1]) ** 2) + convergence_paramater * sum(np.array(theta_params)**2)

    return objective_value


def jacobian_generalized(a, X, y, yl, weight_methods):

    '''
    Returns the Jacobian matrix given the current estimates.

            Parameters:
                    a: The current OLS estimates of X, the current polynomial weights, and possibly the current ylag OLS estimate
                    X: The high-frequency data set
                    y: The dependent variable (growth rate of GDP)
                    yl: The first lag of the dependent variable (growth rate of GDP)
                    weight_methods: The polynomial weighting methods that are used

            Returns:
                    jacobian: The Jacobian matrix
    '''

    # Number of regressors is equal to number of weighting methods
    num_regressors = len(weight_methods)

    # Initialisation a list containing the different elements of the Jacobian
    jwx_all = []

    # Initialisation of weighted regressors
    weighted_regressors = []

    # For all regressors their Jacobian part is calculated
    for i in range(num_regressors):
        theta_start = 1 + i * 2 + num_regressors
        theta_end = theta_start + 2
        jwx = jacobian_wx(X[:, i*3:(i+1)*3], a[theta_start:theta_end], weight_methods[i])
        jwx_all.append(jwx)
        xw, _ = weight_methods[i].x_weighted(X[:, i*3:(i+1)*3], a[theta_start:theta_end])
        weighted_regressors.append(xw)

    # The final Jacobian is constructed using the Jacobians of X, using potentially that of ylag, and using the current coefficients
    if yl is None:
        jac_e = np.ones((len(weighted_regressors[0]), 1))
        for i in range(num_regressors):
            jac_e = np.concatenate([jac_e, weighted_regressors[i].reshape((len(weighted_regressors[i]), 1))], axis=1)
        for i in range(num_regressors):
            jac_e = np.concatenate([jac_e, (a[1 + i] * jwx_all[i])], axis=1)
        jacobian = -1.0 * jac_e
    else:
        jac_e = np.ones((len(weighted_regressors[0]), 1))
        for i in range(num_regressors):
            jac_e = np.concatenate([jac_e, weighted_regressors[i].reshape((len(weighted_regressors[i]), 1))], axis=1)
        for i in range(num_regressors):
            jac_e = np.concatenate([jac_e, (a[1 + i] * jwx_all[i])], axis=1)
        jac_e = np.concatenate([jac_e, yl], axis=1)
        jacobian = -1.0 * jac_e

    return jacobian
