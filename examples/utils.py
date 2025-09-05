import numpy as np
import pandas as pd
from scipy.special import expit
import scipy.stats as stats
import io
import contextlib
import warnings

import math
import os
import sys

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri,numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
# Load the ranger package
ranger = importr("ranger")

here = os.path.dirname(__file__)
r_script = os.path.join(here, 'data_causl.R')


warnings.filterwarnings("ignore", message="R is not initialized by the main thread")

# Suppress R output
@contextlib.contextmanager
def suppress_r_output():
    r_output = io.StringIO()
    with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
        yield

def generate_data_causl(n=10000, nI = 3, nX= 1, nO = 1, nS = 1, ate = 2, beta_cov = 0, strength_instr = 3, strength_conf = 1, strength_outcome = 1, binary_intervention=True):
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        # get the folder containing utils.py
    
        robjects.r['source'](r_script)
        generate_data_causl = robjects.globalenv['data.causl']
        r_dataframe = generate_data_causl(n=n, nI=nI, nX=nX, nO=nO, nS=nS, ate=ate, beta_cov=beta_cov, strength_instr=strength_instr, strength_conf=strength_conf, strength_outcome=strength_outcome, binary_intervention=binary_intervention)
    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)
    return df


def generate_data_longitudinl(n=10000, T=10, random_seed=1024, C_coeff=0):
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r_script)
        generate_data_longitudinl = robjects.globalenv['data.longitudinl']
        r_dataframe = generate_data_longitudinl(n=n, T=T, random_seed=random_seed, C_coeff=C_coeff)
    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)
    # Drop non-feature columns
    columns_to_drop = ['id', 'status', 'T']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    time_steps = T 
    # Extract baseline  covariates (s)
    s_cols = ['C']
    s = df[s_cols].values  # Shape: (n, s_dim)

    # Initialize lists to hold x, z, y for all time steps
    x_list = []
    z_list = []
    y_list = []

    for t in range(time_steps):
        x_col = f"X_{t}"
        z_col = f"Z_{t}"
        y_col = f"Y_{t}"

        if x_col not in df.columns or z_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Expected columns {x_col}, {z_col}, {y_col} not found in the dataframe.")

        x_list.append(df[x_col].values.reshape(-1, 1))  
        z_list.append(df[z_col].values.reshape(-1, 1))  
        y_list.append(df[y_col].values.reshape(-1, 1)) 

    # Concatenate along the second dimension to form [n, T * x_dim], etc.
    x_array = np.concatenate(x_list, axis=1)  # Shape: [n, T * x_dim]
    z_array = np.concatenate(z_list, axis=1)  # Shape: [n, T * z_dim]
    y_array = np.concatenate(y_list, axis=1)  # Shape: [n, T * y_dim]

    return s, x_array, z_array, y_array

def generate_data_survivl(n=10000, T=10, random_seed=1024, C_coeff=0, setting = 1):
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r_script)
        generate_data_survivl = robjects.globalenv['data.survivl']
        r_dataframe = generate_data_survivl(n=n, T=T, random_seed=random_seed, C_coeff=C_coeff, setting = setting)
    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)
    # Drop non-feature columns
    columns_to_drop = ['id', 'status', 'T']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    time_steps = T 
    # Extract baseline  covariates (s)
    s_cols = ['C']
    s = df[s_cols].values  # Shape: (n, s_dim)

    # Initialize lists to hold x, z, y for all time steps
    x_list = []
    z_list = []
    y_list = []

    for t in range(time_steps):
        x_col = f"X_{t}"
        z_col = f"Z_{t}"
        y_col = f"Y_{t}"

        if x_col not in df.columns or z_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Expected columns {x_col}, {z_col}, {y_col} not found in the dataframe.")

        x_list.append(df[x_col].values.reshape(-1, 1))  # Assuming x_dim=1
        z_list.append(df[z_col].values.reshape(-1, 1))  # Assuming z_dim=1
        y_list.append(df[y_col].values.reshape(-1, 1))  # Assuming y_dim=1

    # Concatenate along the second dimension to form [n, T * x_dim], etc.
    x_array = np.concatenate(x_list, axis=1)  # Shape: [n, T * x_dim]
    z_array = np.concatenate(z_list, axis=1)  # Shape: [n, T * z_dim]
    y_array = np.concatenate(y_list, axis=1)  # Shape: [n, T * y_dim]

    return s, x_array, z_array, y_array

