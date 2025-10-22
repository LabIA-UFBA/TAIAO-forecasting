import timesfm
import torch
import pandas as pd
import numpy as np
from dtw import *
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error
)
from typing import Union
def standard_normalize(
    data: Union[torch.Tensor, np.ndarray, list],
    dim: int = 1,
    eps: float = 1e-8,
    return_type: str = "auto"
):
    """
    Normaliza os dados para média 0 e desvio padrão 1 (z-score)
    ao longo da dimensão especificada.

    Args:
        data (torch.Tensor | np.ndarray | list): Dados de entrada.
        dim (int): Dimensão ao longo da qual normalizar (default=1).
        eps (float): Pequeno valor para evitar divisão por zero.
        return_type (str): Tipo de retorno. Pode ser:
            - "auto": retorna no mesmo tipo da entrada
            - "torch": sempre retorna torch.Tensor
            - "numpy": sempre retorna numpy.ndarray

    Returns:
        torch.Tensor | np.ndarray: Dados normalizados (z-score).
    """
    # Detecta o tipo de entrada
    input_type = "torch" if isinstance(data, torch.Tensor) else \
                 "numpy" if isinstance(data, np.ndarray) else \
                 "list"

    # Converte para tensor
    tensor = torch.as_tensor(data, dtype=torch.float32)

    # Normalização z-score
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True, unbiased=False)
    normed = (tensor - mean) / (std + eps)

    # Decide formato de saída
    if return_type == "torch" or (return_type == "auto" and input_type == "torch"):
        return normed
    elif return_type == "numpy" or (return_type == "auto" and input_type in {"numpy", "list"}):
        return normed.numpy()
    else:
        raise ValueError(f"return_type '{return_type}' não suportado.")


# Data loading
df = pd.read_parquet('/home/marcos/workdir/phd/Waikato_TAIAO/dataset_temperature_c_interpolation.parquet')
df.head()

# Get all station columns (exclude 'time' and 'split')
stations = [col for col in df.columns if col not in ['time', 'split']]

# Model configurations
model_configs = [
    {
        'name': '200m',
        'hparams': {
            'backend': 'gpu',
            'per_core_batch_size': 20,
            'horizon_len': 100,
        },
        'checkpoint': timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
        'results_file': 'timesfm_200_results.csv'
    },
    {
        'name': '500m',
        'hparams': {
            'backend': 'gpu',
            'per_core_batch_size': 20,
            'horizon_len': 100,
            'num_layers': 50,
            'use_positional_embedding': False,
            'context_len': 2048,
        },
        'checkpoint': timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
        'results_file': 'timesfm_500_results.csv'
    }
]

# Loop through models
for model_config in model_configs:
    print(f"\n{'='*60}")
    print(f"Running with {model_config['name']} model")
    print(f"{'='*60}\n")
    
    # Initialize model
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(**model_config['hparams']),
        checkpoint=model_config['checkpoint']
    )
    
    all_results = []
    
    # Loop through all stations
    for station in stations:
        print(f"\nProcessing station: {station}")
        
        all_results = []  # Reset results for each station
        
        train_data = df.query("split == True")[station].values[-1000:]
        test_data = df.query("split == False")[station].values[:500]
        
        # Part II - Iterative forecasting
        ini = 0
        end = 100
        features = train_data.copy()
        trues = []
        preds = []
        
        for i in range(4):
            x_test = np.concatenate([features, test_data[ini:end]])
            frequency_input = [0]
            point_forecast, experimental_quantile_forecast = tfm.forecast(
                [x_test],
                freq=frequency_input,
            )
            y_pred = point_forecast.reshape(100)
            trues.extend(test_data[ini+100:end+100].tolist())
            preds.extend(y_pred.tolist())
            features = np.concatenate([features, test_data[ini:end]])
            ini += 100
            end += 100
        
        # Calculate metrics for this station
        y_true = np.array(trues)
        y_pred = np.array(preds)
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        alignment = dtw(y_true, y_pred, keep_internals=True)
        
        print(f"  DTW ....: {alignment.normalizedDistance:.4f}")
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        summary_row = pd.DataFrame([
        {
            "station": station,
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
            "medae": float(medae),
            "evs": float(evs),
        }
        ])
        try:
            existing = pd.read_csv(model_config['results_file'])
            combined = pd.concat([existing, summary_row], ignore_index=True)
        except FileNotFoundError:
            combined = summary_row
        # Drop duplicates on station, keeping the most recent
        combined = combined.drop_duplicates(subset=["station"], keep="last")
        combined.to_csv(model_config['results_file'], index=False)
        print(f"  Metrics summary of {station} saved to {model_config['results_file']}")
        # Store results for this station
        station_results = [
            {'y_true': true_val, 'y_hat': pred_val}
            for true_val, pred_val in zip(trues, preds)
        ]
        
        # Save results to CSV
        results_df = pd.DataFrame(station_results)
        results_filename = f"log_timesfm_{model_config['name']}_{station}.csv"
        results_df.to_csv(results_filename, index=False)
        print(f"  Prediction results saved to {results_filename}")
