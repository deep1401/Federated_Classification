from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import flwr as fl
from handcrafted_GRU import GRU

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


def load_data(inputs, labels, pct_test=0.2):
    train_labels = labels[:-int(len(labels) * pct_test)]
    train_inputs = inputs[:-int(len(labels) * pct_test)]

    test_labels = labels[-int(len(labels) * pct_test):]
    test_inputs = inputs[-int(len(labels) * pct_test):]

    train_set = [(x, y) for x, y in zip(train_inputs, train_labels)]
    test_set = [(x, y) for x, y in zip(test_inputs, test_labels)]

    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = DataLoader(test_set, batch_size=32, shuffle=True)
    return trainloader, testloader


def train(net, trainloader, epochs):
    net.train()
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(epochs):
        for inputs, labels in trainloader:
            h = torch.Tensor(np.zeros((32, HIDDEN_DIM)))
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            output, _ = net(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()


def test(net, testloader):
    net.eval()
    criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            h = torch.Tensor(np.zeros((32, HIDDEN_DIM)))
            inputs, labels = inputs.to(device), labels.to(device)
            output, _ = net(inputs, h)
            loss += criterion(output.squeeze(), labels.float()).item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

    accuracy = correct / total
    return loss, accuracy


net = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)

trainloader, testloader = load_data(inputs, labels)


class FedClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return len(testloader), float(loss), float(accuracy)


fl.client.start_numpy_client("[::]:8080", client=FedClient())
