import os
from preprocess import N_fold, feature_names, input_path, skip_dates, num_valid_dates, TRAINING
from reduce_mem import reduce_mem_usage
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import pandas as pd

if TRAINING:
    df = pd.read_parquet(f'{input_path}/train.parquet')
    
    df = reduce_mem_usage(df, False)
    
    df = df[df['date_id'] >= skip_dates].reset_index(drop=True)
    
    dates = df['date_id'].unique()
    
    valid_dates = dates[-num_valid_dates:]
    
    train_dates = dates[:-num_valid_dates]
    
    print(df.tail())
    
# Create a directory to store the trained models
os.system('mkdir models')

# Define the path to load pre-trained models (if not in training mode)
model_path = '/kaggle/input/jsbaselinezyz'

# If in training mode, prepare validation data
if TRAINING:
    # Extract features, target, and weights for validation dates
    X_valid = df[feature_names].loc[df['date_id'].isin(valid_dates)]
    y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)]
    w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)]

# Initialize a list to store trained models
models = []

# Function to train a model or load a pre-trained model
def train(model_dict, model_name='lgb'):
    if TRAINING:
        # Select dates for training based on the fold number
        selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_fold != i]
        
        # Get the model from the dictionary
        model = model_dict[model_name]
        
        # Extract features, target, and weights for the selected training dates
        X_train = df[feature_names].loc[df['date_id'].isin(selected_dates)]
        y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)]
        w_train = df['weight'].loc[df['date_id'].isin(selected_dates)]

        # Train the model based on the type (LightGBM, XGBoost, or CatBoost)
        if model_name == 'lgb':
            # Train LightGBM model with early stopping and evaluation logging
            model.fit(X_train, y_train, w_train,  
                      eval_metric=[r2_lgb],
                      eval_set=[(X_valid, y_valid, w_valid)], 
                      callbacks=[
                          lgb.early_stopping(100), 
                          lgb.log_evaluation(10)
                      ])
            
        elif model_name == 'cbt':
            # Prepare evaluation set for CatBoost
            evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
            
            # Train CatBoost model with early stopping and verbose logging
            model.fit(X_train, y_train, sample_weight=w_train, 
                      eval_set=[evalset], 
                      verbose=10, 
                      early_stopping_rounds=100)
            
        else:
            # Train XGBoost model with early stopping and verbose logging
            model.fit(X_train, y_train, sample_weight=w_train, 
                      eval_set=[(X_valid, y_valid)], 
                      sample_weight_eval_set=[w_valid], 
                      verbose=10, 
                      early_stopping_rounds=100)

        # Append the trained model to the list
        models.append(model)
        
        # Save the trained model to a file
        joblib.dump(model, f'./models/{model_name}_{i}.model')
        
        # Delete training data to free up memory
        del X_train
        del y_train
        del w_train
        
        # Collect garbage to free up memory
        import gc
        gc.collect()
        
    else:
        # If not in training mode, load the pre-trained model from the specified path
        models.append(joblib.load(f'{model_path}/{model_name}_{i}.model'))
        
    return 


model_dict = {
    'lgb': lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(n_estimators=2000, learning_rate=0.1, max_depth=6, tree_method='hist', device="cuda", objective='reg:squarederror', eval_metric=r2_xgb, disable_default_eval_metric=True),
    'cbt': cbt.CatBoostRegressor(iterations=1000, learning_rate=0.05, task_type='GPU', loss_function='RMSE', eval_metric=r2_cbt()),
}

# Train models for each fold
for i in range(N_fold):
    train(model_dict, 'lgb')
    train(model_dict, 'xgb')
    train(model_dict, 'cbt')