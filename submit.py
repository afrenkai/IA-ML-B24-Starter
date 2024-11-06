import kaggle_evaluation.jane_street_inference_server
from pred import predict
import os

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/test.parquet',
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/lags.parquet',
        )
    )