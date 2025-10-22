# basic
import os
import warnings
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
# pre processing
from sklearn import preprocessing as pre
# NN
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
# val and plot
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error
)
from loguru import logger as log
# plot
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torchviz import make_dot
from tqdm.notebook import trange  # opcional, pra barra de progresso

from torch_geometric.nn import SAGEConv, LEConv, GlobalAttention
from torch_geometric.data import Batch

warnings.filterwarnings("ignore")
import networkx as nx
import torch_geometric.nn as gnn 


# set seed
SEED = 1345
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)
plt.style.use('ggplot')
pd.set_option('display.float_format', '{:.16f}'.format)


def load_dataset(fpath):
    # Load the StaticGraphTemporalSignal object from the file
    with open(fpath, 'rb') as f:
        loaded_temporal_signal = pickle.load(f)
    return loaded_temporal_signal


def load_data(c=51):
    train_dataset = load_dataset(f'dataset_train_{c}_time.pkl')
    test_dataset = load_dataset(f'dataset_test_{c}_time.pkl')
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


class SimpleGraphModel(nn.Module):
    def __init__(self, node_features, hidden_size=64, horizon=100, num_targets=3):
        super().__init__()
        # GCN: uma única camada
        self.conv1 = SAGEConv(node_features, hidden_size)

        # Linear final para previsão
        self.linear = nn.Linear(hidden_size, num_targets * horizon)

        self.horizon = horizon
        self.num_targets = num_targets

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN
        x = self.conv1(x, edge_index).relu()

        # Pooling global por grafo no batch
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_size]

        # Camada final de previsão
        x = self.linear(x)  # [batch_size, num_targets * horizon]

        # Reformatar saída
        x = x.view(-1, self.horizon)  # [batch_size * num_targets, horizon]

        return x
    
    
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        #print(out.shape, batch.y.shape)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def plot_predictions(model, loader, device, num_plots=3):
    model.eval()
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_plots:
                break
            batch = batch.to(device)
            out = model(batch)  # Shape: [batch_size * num_targets, horizon]
            
            # Reformatar para [batch_size, num_targets, horizon]
            out = out.view(-1, model.num_targets, model.horizon)
            true = batch.y.view(-1, model.num_targets, model.horizon)
            
            # Pegar o primeiro grafo do batch
            true = true[0].cpu().numpy()  # Shape: [num_targets, horizon]
            pred = out[0].cpu().numpy()
            
            for target in range(true.shape[0]):
                axes[target].plot(true[target], label='True', color='blue')
                axes[target].plot(pred[target], label='Pred', color='red', linestyle='--')
                axes[target].set_title(f'Target {target + 1}')
                axes[target].legend()
    plt.tight_layout()
    plt.show()


def get_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)     # [51, 100]
            y_true = batch.y       # [51, 100]

            preds.append(out.cpu().numpy())
            trues.append(y_true.cpu().numpy())

    # stack → mantém eixos [janelas, nós, horizonte]
    preds = np.stack(preds, axis=0)
    trues = np.stack(trues, axis=0)

    # print(f"Predictions shape: {preds.shape}")
    # print(f"True values shape: {trues.shape}")

    return preds, trues


if __name__ == "__main__":
    
    c = 51
    device = 'cuda:1'
    
    train_loader, test_loader = load_data()
    
    model = SimpleGraphModel(
        node_features=100,  # conforme seu x=[12, 100]
        horizon=100,
        num_targets=1
    ).to(device)
    
    log.info(f"Train in device: {device}")


    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    # Treinamento
    for epoch in range(1, 1000):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = test(model, test_loader, criterion, device)

        if epoch % 100 == 0:
            log.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            #plot_predictions(model, test_loader, device, num_plots=3)
            
    log.info("Testing")
    # inference
    # Uso:
    predictions, y_trues = get_predictions(model, test_loader, device)
    log.info("Predictions shape:", predictions.shape)  # Ex: (100, 3, 100)
    
    y_pred_src = predictions.copy()
    y_true_src = y_trues.copy()
    
    g = nx.read_graphml("../grafo.graphml")

    #A = nx.adjacency_matrix(g)
    nodes = sorted(g.nodes())  # ou outra lista ordenada de nós
    
    for node in range(51):
        idx = 0
        ytrues = []
        ypreds = []

        for i in range(4):
            y_true = y_true_src[idx, node,:].tolist()
            y_pred = y_pred_src[idx, node,:].tolist()

            ytrues.extend(y_true)
            ypreds.extend(y_pred)

            idx += 100
            if idx == 100:
                idx -= 1
                
        # Cria um DataFrame com previsões
        forecast_df = pd.DataFrame({
            "serie": nodes[node],
            "forecast_mean": ypreds,
            "real": ytrues
        })
        forecast_df.to_csv(f"sage/forecast_{nodes[node]}.csv", index=False)

