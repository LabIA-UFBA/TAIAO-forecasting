import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
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

def exp_worflow(df: pd.DataFrame, time_series: list):

    for ts_same in time_series:
        train_data = df.query("split == True")[ts_same].values[-1000:]
        test_data  = df.query(" split == False ")[ts_same].values[:500]

        
        model_auto = pm.auto_arima(train_data, seasonal=True, 
                                m=30, 
                                trace=True)
        log.info(f"fit {ts_same}")
        log.info(model_auto.summary())
        log.info("")
        
        order = model_auto.order
        seasonal_order = model_auto.seasonal_order


        model = SARIMAX(train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=len(test_data))
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        forecast = results.get_forecast(steps=len(test_data))
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Corrigir os índices para visualização (já que usamos arrays)
        train_idx = np.arange(len(train_data))
        test_idx = np.arange(len(train_data), len(train_data) + len(test_data))

        # Salvar o modelo ajustado
        results.save(f"SARIMA_{ts_same}.pkl")

        # part II

        ini = 0
        end = 100
        features = train_data.copy()

        trues = []
        preds = []

        for i in range(4):

            features = np.concatenate([features, test_data[ini:end]])

            model = SARIMAX(features,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            results = model.fit(disp=False)

            forecast = results.get_forecast(steps=100)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            trues.extend(test_data[ini+100:end+100].tolist())
            preds.extend(forecast_mean)
            
            ini += 100
            end += 100

        # Cria um DataFrame com previsões
        forecast_df = pd.DataFrame({
            "serie": ts_same,
            "forecast_mean": preds,
            "real": trues
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
    