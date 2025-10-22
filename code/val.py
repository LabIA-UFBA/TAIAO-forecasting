import numpy as np

def dtw_distance(a, b):
    """
    Dynamic Time Warping (DTW) entre duas sequências 1D.
    Custo base: |a_i - b_j|.
    Complexidade: O(n*m).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n, m = len(a), len(b)

    # matriz de custos acumulados
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j],     # inserção
                                 D[i, j - 1],     # deleção
                                 D[i - 1, j - 1]) # match
    return D[n, m]


def regression_metrics(y_true, y_pred):
    """
    Calcula DTW, MAE, MSE, RMSE, MAPE (%) e R2 entre y_true e y_pred.

    - Se os comprimentos diferirem, DTW é calculado normalmente, mas
      as métricas ponto-a-ponto (MAE, MSE, RMSE, MAPE, R2) exigem
      o mesmo tamanho e gerarão erro.
    - MAPE ignora elementos com y_true == 0; se todos forem zero, retorna NaN.
    - R2 retorna NaN se a variância de y_true for zero.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # DTW funciona mesmo com comprimentos diferentes
    dtw = dtw_distance(y_true, y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Para MAE, MSE, RMSE, MAPE e R2, y_true e y_pred precisam ter o mesmo tamanho.")

    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)

    nonzero = y_true != 0
    mape = np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0 if np.any(nonzero) else np.nan

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "DTW": float(dtw),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }
