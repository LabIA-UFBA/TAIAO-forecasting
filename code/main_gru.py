import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from loguru import logger as log

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error
)


def create_dataset(series, window_size, horizon):
    """
    Gera X e y para previsão de múltiplos passos.

    Args:
        series (array-like): sequência de valores.
        window_size (int): tamanho da janela de entrada.
        horizon (int): número de passos à frente a prever.

    Returns:
        X: Tensor de forma (n_amostras, window_size)
        y: Tensor de forma (n_amostras, horizon)
    """
    X, y = [], []
    max_start = len(series) - window_size - horizon + 1
    for i in range(max_start):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + horizon])
        
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_dim=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.gru(x) 
        #print(f"out: {out.shape}")
        out = self.linear(out[:, -1, :])  # pegar o último passo da sequência
        #out = self.linear(out[:, -1])  # pegar o último passo da sequência
        
        return out

def get_model(
        window_size = 100,  # tamanho do meu X, 
        hidden_size = 64,   # Camadas ocultas
        n_steps=100,        # horizonte de previsão
        epochs = 1000,       # number of epochs
        lr = 0.001,         # learning rate
        device="cpu",
):
    # model
    model = GRUModel(input_size=1, 
                    hidden_size=hidden_size, 
                    num_layers=2, 
                    output_dim=n_steps).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def exp_worflow(df: pd.DataFrame, 
                time_series: list, 
                window_size = 100, 
                n_steps=100, 
                EPOCHS=1000
):

    for ts_same in time_series:
        train_data = df.query("split == True")[ts_same].values[-1000:]
        test_data  = df.query(" split == False ")[ts_same].values[:500]

        X_train, y_train = create_dataset(train_data, window_size, n_steps)
        X_test, y_test = create_dataset(test_data, window_size, n_steps)

        X_train = X_train.unsqueeze(-1)
        X_test  = X_test.unsqueeze(-1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train = X_train.to(device=device, dtype=torch.float32, non_blocking=True)
        y_train = y_train.to(device=device, dtype=torch.float32, non_blocking=True)
        X_test = X_test.to(device=device, dtype=torch.float32, non_blocking=True)
        y_test = y_test.to(device=device, dtype=torch.float32, non_blocking=True)

        model, criterion, optimizer = get_model(device=device)

        # Loop de treino
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                log.info(f"Epoch {epoch} - Loss: {loss.item():.4f}")
                

        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            test_loss = criterion(preds, y_test)
            print(f"Test Loss: {test_loss.item():.4f}")


        idx = 0

        ytrues = []
        ypreds = []

        for i in range(4):
            
            y_true = y_test[idx,:].cpu().numpy().tolist()
            y_pred = preds[idx,:].cpu().numpy().tolist()
            
            ytrues.extend(y_true)
            ypreds.extend(y_pred)
            
            idx += 100

        # Cria um DataFrame com previsões
        forecast_df = pd.DataFrame({
            "serie": ts_same,
            "forecast_mean": ypreds,
            "real": ytrues
        })

        # Salvar em CSV
        forecast_df.to_csv(f"forecast_{ts_same}.csv", index=False)




if __name__ == '__main__':
    
    df = pd.read_parquet("../../dataset_temperature_c_interpolation.parquet")

    time_series = ['Napier_Aerodrome', 'Wellington_Aerodrome', 'Enderby_Island',
       'Mokohinau_Island', 'Cape_Campbell', 'Westport_Aerodrome',
       'Mahia_NZMHX', 'Nelson_Aerodrome', 'Queenstown_Aerodrome',
       'Auckland_Aerodrome', 'Whanganui_Aerodrome', 'Farewell_Spit',
       'Kaikoura_NZKIX', 'Kerikeri_Aerodrome', 'Castlepoint_NZCPX',
       'Haast_NZHTX', 'New_Plymouth', 'Milford_Sound', 'Whangarei_Aerodrome',
       'Christchurch_Aerodrome', 'Tara_Hills', 'Whitianga_Aerodrome',
       'Paraparaumu_Aerodrome', 'Le_Bons', 'Waiouru_Aerodrome', 'Ngawi_NZNWX',
       'Oamaru_Aerodrome', 'Hokitika_Aerodrome', 'Stephens_Island',
       'Tauranga_Aerodrome', 'Raoul_Island', 'Hamilton_Aerodrome', 'Hicks_Bay',
       'Timaru_Airport', 'Taupo_Aerodrome', 'Puysegur_Point', 'Chatham_Island',
       'South_West', 'Hawera_NZHAX', 'Mt_Cook', 'Campbell_Island',
       'Invercargill_NZNVA', 'Port_Taharoa', 'Palmerston_N',
       'Gisborne_Aerodrome', 'Nugget_Point', 'Dunedin_Aerodrome',
       'Cape_Reinga', 'Secretary_Island', 'Kaitaia_Aerodrome',
       'Takapau_Plains']
    
    exp_worflow(df, time_series)
    