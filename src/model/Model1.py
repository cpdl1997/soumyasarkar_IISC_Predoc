import torch
import torch.nn as nn

class Model1(torch.nn.Module):

    def __init__(self, _in_features, _out_features, _learning_rate=0.01):
        super().__init__()
        self._in_features = _in_features
        self._out_features = _out_features
        self._learning_rate = _learning_rate
        self.lin1 = nn.Linear(in_features = _in_features, out_features = 60)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features = 60, out_features = 30)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features = 30, out_features = _out_features+1)
        self.act3 = nn.ReLU()
        self.act4 = nn.Softmax()
        
    
    def forward(self, train):
        x = self.act1(self.lin1(train))
        x = self.act2(self.lin2(x))
        x1 = self.act4(self.lin3(x)[:,0:self._output_size])
        x2 = self.act3(self.lin3(x)[:,self._output_size:self._output_size+1])
        output = torch.concat([x1,x2], axis=1)
        return output
