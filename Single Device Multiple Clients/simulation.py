from multiprocessing import Process
from collections import OrderedDict
import os
from typing import Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from handcrafted_GRU import GRU
import dataset

# Data files
inputs = torch.tensor(np.load('../data/inputs.npy'))
labels = torch.tensor(np.load('../data/labels.npy'))

# Model params
EMBEDDING_DIM = 50
HIDDEN_DIM = 10
DROPOUT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = int(inputs.max()) + 1
CLIP = 5
DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def start_server(num_rounds:int, num_clients:int, fraction_fit: float):
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    fl.server.start_server(strategy=strategy, config={'num_rounds':num_rounds})


def train(net, train_loader, epochs):
    net.train()
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(epochs):
        for inputs, labels in train_loader:
            h = torch.Tensor(np.zeros((32, HIDDEN_DIM)))
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            output, _ = net(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()


def test(net, test_loader):
    net.eval()
    criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            h = torch.Tensor(np.zeros((32, HIDDEN_DIM)))
            inputs, labels = inputs.to(device), labels.to(device)
            output, _ = net(inputs, h)
            loss += criterion(output.squeeze(), labels.float()).item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

    accuracy = correct / total
    return loss, accuracy


def start_client(dataset : DATASET):
    """Start a client instance"""
    net = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)
    (x_train, y_train), (x_test, y_test) = dataset
    train_loader = DataLoader([(x,y) for x,y in zip(x_train,y_train)], batch_size=32, shuffle=True)
    test_loader = DataLoader([(x,y) for x,y in zip(x_test,y_test)], batch_size=32, shuffle=True)

    class FedClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, train_loader, epochs=1)
            return self.get_parameters(), len(train_loader)

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, test_loader)
            return len(test_loader), float(loss), float(accuracy)

    fl.client.start_numpy_client("0.0.0.0:8080", client=FedClient())


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):

    # Hold all processes
    processes = []

    server_process = Process(target=start_server, args=(num_rounds, num_clients, fraction_fit))
    server_process.start()
    processes.append(server_process)

    time.sleep(2)

    partitions = dataset.load_data(num_partitions=num_clients)

    for partition in partitions:
        client_process = Process(target=start_client, args=(partition,))
        client_process.start()
        processes.append(client_process)

    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation(num_rounds=100, num_clients=10, fraction_fit=0.5)






