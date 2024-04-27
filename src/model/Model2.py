import torch
import torch.nn as nn

class Model2Genre(torch.nn.Module):

    def __init__(self, _in_features, _out_features, _learning_rate=0.01):
        super().__init__()
        self._in_features = _in_features
        self._out_features = _out_features
        self._learning_rate = _learning_rate
        self.lin1 = nn.Linear(in_features = _in_features, out_features = 60)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features = 60, out_features = 30)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features = 30, out_features = _out_features)
        
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = torch.exp(x/1e6)
        return e_x/torch.sum(e_x)

    
    def forward(self, train):
        x = self.act1(self.lin1(train))
        x = self.act2(self.lin2(x))
        x = self.softmax(self.lin3(x))
        return x
    


class Model2Year(torch.nn.Module):

    def __init__(self, _in_features, _learning_rate=0.01):
        super().__init__()
        self._in_features = _in_features
        self._learning_rate = _learning_rate
        self.lin1 = nn.Linear(in_features = _in_features, out_features = 60)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features = 60, out_features = 30)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features = 30, out_features = 1)
        self.act3 = nn.ReLU()

    
    def forward(self, train):
        x = self.act1(self.lin1(train))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        return x
