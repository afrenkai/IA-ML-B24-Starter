import os
from reduce_mem import reduce_mem_usage
import pandas as pd

input_path = './jane-street-real-time-market-data-forecasting/' if os.path.exists('./jane-street-real-time-market-data-forecasting') else '/kaggle/input/jane-street-real-time-market-data-forecasting/'

TRAINING = False

feature_names = [f"feature_{i:02d}" for i in range(79)]

num_valid_dates = 100

skip_dates = 500

N_fold = 5

