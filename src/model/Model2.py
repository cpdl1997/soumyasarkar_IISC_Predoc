import torch
import torch.nn as nn
from torch.nn.modules import Module

class Model2Genre(torch.nn.Module):

    # def __init__(self, _in_features, _out_features, _learning_rate=0.01):
    #     super().__init__()
    #     self._in_features = _in_features
    #     self._out_features = _out_features
    #     self._learning_rate = _learning_rate
    #     self.lin1 = nn.Linear(in_features = _in_features, out_features = 60)
    #     self.act1 = nn.ReLU()
    #     self.lin2 = nn.Linear(in_features = 60, out_features = 30)
    #     self.act2 = nn.ReLU()
    #     self.lin3 = nn.Linear(in_features = 30, out_features = _out_features)
    #
    # def forward(self, train):
    #     x = self.act1(self.lin1(train))
    #     x = self.act2(self.lin2(x))
    #     x = self.softmax(self.lin3(x))
    #     return x

    #https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124/6
    def __init__(self, _in_features, _out_features, hidden_layers_data: list, _learning_rate=0.01):
        super().__init__()
        self.input_size = _in_features
        self.learning_rate = _learning_rate
        self.layers = nn.ModuleList()

        for size, activation in hidden_layers_data:
            self.layers.append(nn.Linear(_in_features, size))
            _in_features = size  # For the next layer
            if activation is not None:
                self.layers.append(activation)

        self.layers.append(nn.Linear(_in_features, _out_features)) #final layer

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)
        
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        input_data = self.softmax(input_data)
        return input_data
        
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = torch.exp(x/1e6)
        return e_x/torch.sum(e_x)
    
    


class Model2Year(torch.nn.Module):

    # def __init__(self, _in_features, _learning_rate=0.01):
    #     super().__init__()
    #     self._in_features = _in_features
    #     self._learning_rate = _learning_rate
    #     self.lin1 = nn.Linear(in_features = _in_features, out_features = 60)
    #     self.act1 = nn.ReLU()
    #     self.lin2 = nn.Linear(in_features = 60, out_features = 30)
    #     self.act2 = nn.ReLU()
    #     self.lin3 = nn.Linear(in_features = 30, out_features = 1)
    #     self.act3 = nn.ReLU()

    
    # def forward(self, train):
    #     x = self.act1(self.lin1(train))
    #     x = self.act2(self.lin2(x))
    #     x = self.act3(self.lin3(x))
    #     return x

    #https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124/6
    def __init__(self, _in_features, hidden_layers_data: list, _learning_rate=0.01):
        super().__init__()
        self.input_size = _in_features
        self.learning_rate = _learning_rate
        self.layers = nn.ModuleList()

        for size, activation in hidden_layers_data:
            self.layers.append(nn.Linear(_in_features, size))
            _in_features = size  # For the next layer
            if activation is not None:
                self.layers.append(activation)

        self.layers.append(nn.Linear(_in_features, 1)) #final layer
        self.layers.append(nn.ReLU())

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)
        
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
