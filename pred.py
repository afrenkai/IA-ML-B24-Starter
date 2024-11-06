import polars as pl
from preprocess import feature_names
import numpy as np
from train import models
import pandas as pd

lags_ : pl.DataFrame | None = None

# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each batch of predictions (except the very first) must be returned within 10 minutes of the batch features being provided.
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    
    feat = test[feature_names].to_numpy()
    
    pred = [model.predict(feat) for model in models]
    pred = np.mean(pred, axis=0)
    
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions